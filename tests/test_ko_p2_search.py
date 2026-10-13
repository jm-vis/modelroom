"""AP9-KO P2: the direct find (V8, L2) and the knowledge hint (V9, L3) of the search of step 2.

Synthetic throughout, after the pattern of the real cases and without any real name as a special
case: a model of a publisher with no GGUF at all, a model that only a foreign account packages, and
the counter-case of a model a listed packager packages. Every transport here is a small fake Hub.
"""

from __future__ import annotations

import http.client
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from modelroom import guided_search
from modelroom.catalog import Catalog, CatalogFamily, CatalogModel
from modelroom.catalog_findings import KNOWLEDGE_HINT_MAX, knowledge_hint_text
from modelroom.config import Configuration, load_config
from modelroom.dialog import FileAsker
from modelroom.guided import GuidedRun, _search_step
from modelroom.guided_contracts import DIRECT_FIND_MAX
from modelroom.http import RequestBudget, Response
from modelroom.local_servers import ServerRun
from modelroom.screen import Screen
from modelroom.search import SearchOutcome, run_search
from modelroom.search_direct import find_direct
from modelroom.search_pages import account_downloads_url, account_search_url, model_url
from modelroom.search_apply import write_configuration
from modelroom.search_word import parse_search
from modelroom.tensor_search import search_tensor_repos, tensor_search_url

from fixture_support import offline_daemon, offline_servers, qwen35_example_config_dict, windows_probes

NOW = datetime(2026, 10, 13, 9, 0, 0, tzinfo=timezone.utc)
HF = "https://huggingface.co/api"

TENSOR_ONLY = "acme/Nova-9B"  # a publisher model, no GGUF anywhere
FOREIGN_ONLY = "kiteai/Kite-1"  # only an account nobody listed packages it
LISTED = "orbitco/Orbit-30B"  # a listed packager packages it (the counter-case)


# --- the synthetic catalog and Hub ---------------------------------------------------------------------------------------


def _model(repo: str, *, latest: bool = False, kind: str = "text", gguf: tuple[str, ...] = (), ollama: str | None = None):
    owner = repo.split("/", 1)[0]
    data: dict = {"hf_repo": repo, "source": f"https://huggingface.co/{repo}", "kind": kind, "gguf_repos": list(gguf)}
    if latest:
        data.update(latest=True, latest_source=f"https://huggingface.co/{owner}", latest_checked=date(2026, 10, 1))
    if ollama is not None:
        data["ollama_base"], _, data["ollama_tag"] = ollama.partition(":")
    return CatalogModel(**data)


def _catalog(extra: list[CatalogFamily] | None = None) -> Catalog:
    families = [
        CatalogFamily(name="nova", publisher="acme", models=[_model(TENSOR_ONLY, latest=True), _model("acme/Nova-3B")]),
        CatalogFamily(name="kite", publisher="kiteai", models=[_model(FOREIGN_ONLY, latest=True)]),
        CatalogFamily(
            name="orbit", publisher="orbitco", models=[_model(LISTED, latest=True, gguf=("unsloth/Orbit-30B-GGUF",))]
        ),
        *(extra or []),
    ]
    return Catalog(schema_version=2, families=families, packagers=["unsloth"])


def _many(prefix: str, count: int) -> CatalogFamily:
    """A family whose models all match the word `zed`: more rows than a direct find or a hint may show."""
    models = [_model(f"zedco/Zed-{index:02d}B", latest=index % 2 == 0) for index in range(1, count + 1)]
    return CatalogFamily(name=prefix, publisher="zedco", models=models)


def _answer(repo: str, **fields) -> Response:
    body = {
        "id": repo,
        "tags": ["safetensors"],
        "cardData": {"license": "apache-2.0"},
        "createdAt": "2026-10-01T10:00:00Z",
        "downloads": 1200,
        "safetensors": {"total": 9_000_000_000},
    }
    body.update(fields)
    return Response(status=200, headers={}, body=json.dumps(body).encode("utf-8"))


def _page(entries: list[dict]) -> Response:
    return Response(status=200, headers={}, body=json.dumps(entries).encode("utf-8"))


class FakeHub:
    """The Hub as far as these tests ask it: model objects by id, list pages by exact url, else an empty list."""

    def __init__(self, models: dict[str, Response] | None = None, pages: dict[str, Response] | None = None) -> None:
        self.models = models or {}
        self.pages = pages or {}
        self.calls: list[str] = []

    def __call__(self, method: str, url: str, headers=None, *, max_bytes=None) -> Response:
        self.calls.append(url)
        if url.startswith(f"{HF}/models/"):
            repo = url[len(f"{HF}/models/") :].split("?", 1)[0]
            return self.models.get(repo, Response(status=404, headers={}, body=b"{}"))
        return self.pages.get(url, _page([]))

    def model_calls(self) -> list[str]:
        return [url for url in self.calls if url.startswith(f"{HF}/models/")]


def _gguf_entry(repo: str, base: str) -> dict:
    return {
        "id": repo,
        "tags": ["gguf", f"base_model:{base}", f"base_model:quantized:{base}"],
        "cardData": None,
        "downloads": 50,
        "createdAt": "2026-09-01T10:00:00Z",
    }


# --- find_direct: the contract of V8 -------------------------------------------------------------------------------------


def _direct(catalog: Catalog, hub, text: str, *, limit: int = 100, skip=frozenset()):
    return find_direct(hub, RequestBudget(limit), catalog, parse_search(text), skip=skip)


def test_a_word_finds_the_publishers_own_model_as_a_direct_find_with_no_invented_relation():
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    outcome = _direct(_catalog(), hub, "nova-9b")

    [hit] = outcome.hits
    assert (hit.repo, hit.found_as, hit.resolved, hit.resolved_base_model) == (TENSOR_ONLY, "direct", True, TENSOR_ONLY)
    assert (hit.publisher_status, hit.base_model, hit.base_model_relation) == ("publisher", None, "unknown")
    assert (hit.downloads, hit.parameters_b, hit.license) == (1200, 9.0, "apache-2.0")
    [entry] = outcome.log
    assert (entry.repo, entry.source, entry.status, entry.requests, entry.reason) == (TENSOR_ONLY, "word", "found", 1, None)
    assert hub.model_calls() == [model_url(TENSOR_ONLY)]


def test_the_age_and_the_ollama_name_of_a_direct_find_come_from_the_catalog():
    catalog = Catalog(
        schema_version=2,
        families=[
            CatalogFamily(
                name="nova",
                publisher="acme",
                models=[
                    _model("acme/Nova-9B", latest=True, ollama="nova:9b"),
                    CatalogModel(hf_repo="acme/Nova-3B", source="https://huggingface.co/acme/Nova-3B", successor="acme/Nova-9B"),
                ],
            )
        ],
    )
    hub = FakeHub({"acme/Nova-9B": _answer("acme/Nova-9B"), "acme/Nova-3B": _answer("acme/Nova-3B")})

    hits = {hit.repo: hit for hit in _direct(catalog, hub, "nova").hits}

    assert (hits["acme/Nova-9B"].age, hits["acme/Nova-9B"].ollama) == ("latest", "nova:9b")
    assert (hits["acme/Nova-3B"].age, hits["acme/Nova-3B"].successor) == ("legacy", "acme/Nova-9B")


def test_a_typed_id_asks_for_the_row_with_that_id_whatever_its_case():
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    outcome = _direct(_catalog(), hub, "ACME/nova-9b")

    assert [hit.repo for hit in outcome.hits] == [TENSOR_ONLY]
    assert outcome.log[0].source == "id"


def test_a_typed_id_that_is_in_no_row_and_a_search_with_no_word_ask_nothing():
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    assert _direct(_catalog(), hub, "stranger/Model-1").log == []
    assert _direct(_catalog(), hub, "").log == []
    assert _direct(_catalog(), hub, "gguf the").log == []  # nothing but filler is no word
    assert hub.calls == []


def test_the_rows_are_the_ones_the_words_name_latest_first_and_only_text_or_vision():
    catalog = Catalog(
        schema_version=2,
        families=[
            CatalogFamily(
                name="nova",
                publisher="acme",
                models=[
                    _model("acme/Nova-old"),
                    _model("acme/Nova-new", latest=True),
                    _model("acme/Nova-pix", kind="image-generation", latest=True),
                    _model("acme/Nova-see", kind="vision"),
                ],
            )
        ],
    )
    hub = FakeHub({repo: _answer(repo) for repo in ("acme/Nova-old", "acme/Nova-new", "acme/Nova-see")})

    outcome = _direct(catalog, hub, "nova")

    assert [hit.repo for hit in outcome.hits] == ["acme/Nova-new", "acme/Nova-old", "acme/Nova-see"]


def test_at_most_the_direct_find_limit_is_asked_and_a_row_a_package_hit_resolved_is_not_asked_again():
    catalog = _catalog([_many("zed", DIRECT_FIND_MAX + 5)])
    hub = FakeHub({model.hf_repo: _answer(model.hf_repo) for f in catalog.families for model in f.models})

    outcome = _direct(catalog, hub, "zed")

    assert len(outcome.log) == len(hub.model_calls()) == DIRECT_FIND_MAX
    skipped = {"zedco/zed-02b"}  # by the casefolded key
    outcome = _direct(catalog, FakeHub({m.hf_repo: _answer(m.hf_repo) for f in catalog.families for m in f.models}), "zed", skip=skipped)
    assert "zedco/Zed-02B" not in [entry.repo for entry in outcome.log]


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (Response(status=404, headers={}, body=b"{}"), "HTTP 404"),
        (Response(status=200, headers={}, body=b"not json"), "the answer is not JSON"),
        (Response(status=200, headers={}, body=b"[]"), "the answer is not a repository"),
        (_answer("other/Model"), "the answer names another repository"),
    ],
)
def test_a_model_that_could_not_be_read_is_named_in_the_log_and_left_out_of_the_hits(response, reason):
    hub = FakeHub({TENSOR_ONLY: response})

    outcome = _direct(_catalog(), hub, "nova-9b")

    assert outcome.hits == []
    [entry] = outcome.log
    assert (entry.repo, entry.status, entry.requests, entry.reason) == (TENSOR_ONLY, "not_read", 1, reason)


def test_a_transport_that_fails_is_a_model_that_was_not_read_and_the_next_one_is_still_asked():
    class Broken(FakeHub):
        def __call__(self, method, url, headers=None, *, max_bytes=None):
            if "Nova-3B" in url:
                raise OSError("connection reset")
            return super().__call__(method, url, headers, max_bytes=max_bytes)

    hub = Broken({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    outcome = _direct(_catalog(), hub, "nova")

    assert [hit.repo for hit in outcome.hits] == [TENSOR_ONLY]
    assert [(entry.repo, entry.status) for entry in outcome.log] == [(TENSOR_ONLY, "found"), ("acme/Nova-3B", "not_read")]
    assert outcome.log[1].reason == "the request failed: OSError"


def test_a_budget_that_ends_names_every_model_it_cut_off_with_no_request():
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY), "acme/Nova-3B": _answer("acme/Nova-3B")})

    outcome = _direct(_catalog(), hub, "nova", limit=1)

    assert [hit.repo for hit in outcome.hits] == [TENSOR_ONLY]
    assert [(entry.repo, entry.status, entry.requests) for entry in outcome.log] == [
        (TENSOR_ONLY, "found", 1),
        ("acme/Nova-3B", "budget", 0),
    ]
    assert outcome.log[1].reason == "the request budget ended"
    assert len(hub.model_calls()) == 1


# --- the step: search_step and the choice (guided._search_step) ----------------------------------------------------------


def _first_config(tmp_path: Path) -> tuple[Path, Configuration]:
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data.update({"schema_version": 2, "families": [], "publishers": []})
    path = tmp_path / "modelroom.toml"
    write_configuration(path, Configuration.from_dict(data), now=NOW, expected_text=None)
    return path, load_config(path)


def _run(tmp_path: Path, hub, catalog: Catalog, answers: dict, lines: list[str], *, vllm: bool, monkeypatch) -> GuidedRun:
    monkeypatch.setattr(guided_search, "vllm_among_chosen", lambda run, config: vllm)
    return GuidedRun(
        asker=FileAsker({"filter_owners": True, **answers}),
        here=tmp_path,
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=hub,
        probes=windows_probes(),
        daemon=offline_daemon(),
        servers=ServerRun(offline_servers()),
        now=NOW,
        catalog=catalog,
        out=lines.append,
        screen=Screen(lines.append, colored=False),
    )


def _search_log(config: Configuration) -> dict:
    return json.loads((config.paths.state / "search.json").read_text(encoding="utf-8"))


def _step(tmp_path, monkeypatch, hub, answers, *, vllm, catalog=None):
    path, config = _first_config(tmp_path)
    lines: list[str] = []
    run = _run(tmp_path, hub, catalog or _catalog(), answers, lines, vllm=vllm, monkeypatch=monkeypatch)
    step = guided_search.search_step(run, config, path.parent)
    return step, run, config, lines


def test_with_a_vllm_machine_and_the_filter_on_a_tensor_only_model_is_in_the_list(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    step, _run_, config, _lines = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=True)

    assert [model.base_model for model in step.models] == [TENSOR_ONLY]
    assert [hit.found_as for hit in step.models[0].repos] == ["direct"]
    assert [entry["repo"] for entry in _search_log(config)["direct"]] == [TENSOR_ONLY]
    assert "knowledge_hint" not in _search_log(config)


def test_a_typed_id_of_a_model_with_no_gguf_is_found_directly(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    step, _run_, config, _lines = _step(tmp_path, monkeypatch, hub, {"search": TENSOR_ONLY}, vllm=True)

    assert [model.base_model for model in step.models] == [TENSOR_ONLY]
    assert [(e["repo"], e["source"], e["status"]) for e in _search_log(config)["direct"]] == [(TENSOR_ONLY, "id", "found")]


def test_the_direct_find_comes_before_the_question_after_a_typo(tmp_path, monkeypatch):
    """The words match a catalog row, so the list has a model and the candidate question is never asked."""
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    step, _run_, _config, _lines = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=True)

    assert step.outcome.candidates == []
    assert step.models


def test_without_a_vllm_machine_no_direct_request_is_made_and_the_gray_line_stands_in(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    step, _run_, config, lines = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=False)

    assert hub.model_calls() == []
    assert step.models == []
    assert any(line.strip().endswith(knowledge_hint_text("Nova-9B", "acme", tensor_offer=False)) for line in lines)
    assert "direct" not in _search_log(config) and "knowledge_hint" not in _search_log(config)


def test_the_knowledge_hint_names_the_tensor_offer_only_on_a_positive_proof_in_the_snapshot(tmp_path, monkeypatch):
    path, config = _first_config(tmp_path)
    monkeypatch.setattr(guided_search, "_tensor_proven", lambda cfg: {"acme/nova-9b"})
    lines: list[str] = []
    run = _run(tmp_path, FakeHub(), _catalog(), {"search": "nova-9b"}, lines, vllm=False, monkeypatch=monkeypatch)

    guided_search.search_step(run, config, path.parent)

    said = " ".join(line.strip() for line in lines)  # the long sentence is broken over lines, its words are not changed
    assert knowledge_hint_text("Nova-9B", "acme", tensor_offer=True) in said
    assert max(len(line) for line in lines if "Nova-9B" in line or "tensor packages" in line) <= 100


def test_the_snapshot_proof_needs_an_active_complete_hugging_face_tensor_package():
    from modelroom.contracts import Package, PackageFile

    def package(repo: str, base: str, *, fmt="tensor", active=True, files=None, complete=True) -> Package:
        weights = files or [PackageFile(name="model.safetensors", role="weights", size_bytes=10)]
        return Package(
            source="huggingface", repo=repo, revision="a" * 40, base_model_hf_repo=base, format=fmt, files=weights,
            complete=complete, provenance="from_metadata", observed_at=NOW, last_seen=NOW, active=active,
        )

    first_of_two = [PackageFile(name="model-00001-of-00002.safetensors", role="weights_shard", size_bytes=10)]
    packages = [
        package("acme/Nova-9B", "Acme/Nova-9B"),
        package("acme/Nova-3B", "acme/Nova-3B", active=False),
        package("acme/Nova-1B", "acme/Nova-1B", fmt="gguf"),
        package("acme/Nova-7B", "acme/Nova-7B", files=first_of_two, complete=False),
    ]

    assert guided_search.proven_tensor_models(packages) == {"acme/nova-9b"}


def test_the_hint_is_limited_and_the_rest_is_counted_in_the_search_log(tmp_path, monkeypatch):
    catalog = _catalog([_many("zed", KNOWLEDGE_HINT_MAX + 4)])

    step, _run_, config, lines = _step(tmp_path, monkeypatch, FakeHub(), {"search": "zed"}, vllm=False, catalog=catalog)

    shown = [line for line in lines if "is in the catalog; no GGUF from its publisher" in line]
    assert len(shown) == KNOWLEDGE_HINT_MAX
    assert _search_log(config)["knowledge_hint"] == 4
    # latest rows first, then the catalog order
    assert "Zed-02B" in shown[0] and "Zed-04B" in shown[1] and "Zed-06B" in shown[2]


def test_a_model_a_listed_packager_packages_gets_no_hint_and_no_direct_find(tmp_path, monkeypatch):
    """The counter-case: the row lists a GGUF repository of a listed packager, so nothing is added."""
    hub = FakeHub()

    _step1, _r, _c, lines = _step(tmp_path, monkeypatch, hub, {"search": "orbit"}, vllm=False)

    assert not any("is in the catalog; no GGUF" in line for line in lines)


def test_a_gguf_repository_this_search_found_counts_as_a_known_gguf(tmp_path, monkeypatch):
    """The row lists none, but a GGUF hit of this very search resolved to the model: no hint."""
    pages = {
        account_search_url("nova", "acme"): _page([_gguf_entry("acme/Nova-9B-GGUF", TENSOR_ONLY)]),
        account_downloads_url("nova", "acme"): _page([]),
    }
    hub = FakeHub(pages=pages)

    _step1, _r, _c, lines = _step(tmp_path, monkeypatch, hub, {"search": "nova"}, vllm=False)

    assert not any("Nova-9B (acme) is in the catalog" in line for line in lines)
    assert any("Nova-3B (acme) is in the catalog" in line for line in lines)


def test_a_model_only_a_foreign_account_packages_is_found_directly_with_the_filter_on(tmp_path, monkeypatch):
    foreign = "otherhub/Kite-1-GGUF"
    pages = {account_search_url("kite", "unsloth"): _page([_gguf_entry(foreign, FOREIGN_ONLY)])}
    hub = FakeHub({FOREIGN_ONLY: _answer(FOREIGN_ONLY)}, pages)

    step, _r, _c, _l = _step(tmp_path, monkeypatch, hub, {"search": "kite"}, vllm=True)

    # the foreign account's repository is no choice with the filter on; the publisher's own model is
    assert [model.base_model for model in step.models] == [FOREIGN_ONLY]
    assert [hit.found_as for hit in step.models[0].repos] == ["direct"]


def test_with_the_filter_off_the_foreign_repository_already_offers_the_model_so_it_is_one_line(tmp_path, monkeypatch):
    """Deduplicated by the base model's id: the model is offered once, and the direct find is not even asked."""
    foreign = "otherhub/Kite-1-GGUF"
    pages = {account_search_url("kite", "unsloth"): _page([_gguf_entry(foreign, FOREIGN_ONLY)])}
    hub = FakeHub({FOREIGN_ONLY: _answer(FOREIGN_ONLY)}, pages)

    step, _r, _c, _l = _step(tmp_path, monkeypatch, hub, {"search": "kite", "filter_owners": False}, vllm=True)

    assert [model.base_model for model in step.models] == [FOREIGN_ONLY]
    assert [hit.found_as for hit in step.models[0].repos] == ["package"]
    assert model_url(FOREIGN_ONLY) not in hub.calls


def test_a_base_model_a_gguf_hit_already_offers_is_not_asked_for_directly(tmp_path, monkeypatch):
    pages = {account_search_url("nova", "acme"): _page([_gguf_entry("acme/Nova-9B-GGUF", TENSOR_ONLY)])}
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)}, pages)

    step, _r, config, _l = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=True)

    assert model_url(TENSOR_ONLY) not in hub.calls
    assert [model.base_model for model in step.models] == [TENSOR_ONLY]
    assert "direct" not in _search_log(config)


# --- choosing: only a chosen model reaches the configuration -------------------------------------------------------------


def _chosen(tmp_path, monkeypatch, hub, answers, *, vllm=True):
    path, config = _first_config(tmp_path)
    lines: list[str] = []
    run = _run(tmp_path, hub, _catalog(), answers, lines, vllm=vllm, monkeypatch=monkeypatch)
    updated, count = _search_step(run, path, config)
    return path, updated, count, lines


def test_a_chosen_direct_find_is_written_with_its_original_repository(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    path, updated, count, _lines = _chosen(tmp_path, monkeypatch, hub, {"search": "nova-9b", "select": [TENSOR_ONLY]})

    assert count == 1
    [family] = updated.families
    [base] = family.base_models
    assert (base.hf_repo, list(base.repos)) == (TENSOR_ONLY, [TENSOR_ONLY])
    assert load_config(path).publishers == ["acme"]


def test_no_choice_is_no_change_to_the_configuration(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})
    path, config = _first_config(tmp_path)
    before = path.read_bytes()
    run = _run(tmp_path, hub, _catalog(), {"search": "nova-9b", "select": []}, [], vllm=True, monkeypatch=monkeypatch)

    updated, count = _search_step(run, path, config)

    assert count == 0 and updated.families == []
    assert path.read_bytes() == before


def test_only_the_chosen_models_are_written_not_every_one_the_search_found(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY), "acme/Nova-3B": _answer("acme/Nova-3B")})

    _path, updated, count, _lines = _chosen(tmp_path, monkeypatch, hub, {"search": "nova", "select": [TENSOR_ONLY]})

    assert count == 1
    assert [base.hf_repo for family in updated.families for base in family.base_models] == [TENSOR_ONLY]


def test_a_search_run_twice_gives_the_same_configuration(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})
    answers = {"search": "nova-9b", "select": [TENSOR_ONLY]}
    path, first, _count, _lines = _chosen(tmp_path, monkeypatch, hub, answers)
    text = path.read_text(encoding="utf-8")

    lines: list[str] = []
    run = _run(tmp_path, hub, _catalog(), answers, lines, vllm=True, monkeypatch=monkeypatch)
    again, count = _search_step(run, path, load_config(path))

    assert count == 1
    assert again.model_dump(mode="json") == first.model_dump(mode="json")
    assert path.read_text(encoding="utf-8") == text


def test_a_change_another_run_made_while_the_list_was_open_is_kept(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    class Racing(FileAsker):
        def checkbox(self, key, question, choices, instruction=None, header=None, notice=None):
            other = load_config(path_of[0])
            data = other.model_dump(mode="json")
            data["publishers"] = [*data["publishers"], "elsewhere"]
            text = path_of[0].read_text(encoding="utf-8")
            write_configuration(path_of[0], Configuration.from_dict(data), now=NOW, expected_text=text)
            return super().checkbox(key, question, choices, instruction)

    path_of: list[Path] = []
    path, config = _first_config(tmp_path)
    path_of.append(path)
    lines: list[str] = []
    run = _run(tmp_path, hub, _catalog(), {}, lines, vllm=True, monkeypatch=monkeypatch)
    run.asker = Racing({"filter_owners": True, "search": "nova-9b", "select": [TENSOR_ONLY]})

    updated, count = _search_step(run, path, config)

    assert count == 1
    assert set(updated.publishers) == {"acme", "elsewhere"}
    assert load_config(path).publishers == updated.publishers


def test_run_search_alone_is_unchanged_for_a_run_that_asks_no_direct_find():
    """The outcome carries nothing new unless a direct find was made: the log keys stay out."""
    hub = FakeHub()

    outcome: SearchOutcome = run_search(hub, "nova", catalog=_catalog(), listed_packagers=["unsloth"], budget=RequestBudget(50))

    log = outcome.search_log(filtered=True, run_at=NOW, unresolved={})
    assert "direct" not in log and "knowledge_hint" not in log
    assert hub.model_calls() == []


# --- fix round 1 (Codex p2-b1) -------------------------------------------------------------------------------------------


def test_a_redirect_is_not_followed_so_every_model_costs_exactly_one_request():
    catalog = _catalog([_many("zed", DIRECT_FIND_MAX + 2)])
    moved = Response(status=301, headers={"Location": f"{HF}/models/elsewhere/Moved"}, body=b"")
    hub = FakeHub({model.hf_repo: moved for f in catalog.families for model in f.models})
    budget = RequestBudget(100)

    outcome = find_direct(hub, budget, catalog, parse_search("zed"))

    assert len(hub.model_calls()) == budget.used == DIRECT_FIND_MAX
    assert {(e.status, e.requests, e.reason) for e in outcome.log} == {("not_read", 1, "HTTP 301")}


def test_the_search_log_books_the_requests_of_the_direct_find_and_the_budget_that_ended_in_it(tmp_path, monkeypatch):
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY), "acme/Nova-3B": _answer("acme/Nova-3B")})
    path, config = _first_config(tmp_path)
    run = _run(tmp_path, hub, _catalog(), {"search": "nova"}, [], vllm=True, monkeypatch=monkeypatch)
    run.budget = RequestBudget(5)  # four account pages, then room for one direct request

    guided_search.search_step(run, config, path.parent)

    log = _search_log(config)
    assert [(e["repo"], e["status"], e["requests"]) for e in log["direct"]] == [(TENSOR_ONLY, "found", 1), ("acme/Nova-3B", "budget", 0)]
    assert log["budget"] == {"used": run.budget.used, "limit": 5}
    assert log["budget"]["used"] == 5


def test_a_transport_failure_of_the_tensor_search_keeps_the_direct_find_and_its_log(tmp_path, monkeypatch):
    class TensorDown(FakeHub):
        def __call__(self, method, url, headers=None, *, max_bytes=None):
            if "filter=safetensors" in url:
                raise OSError("connection reset")
            return super().__call__(method, url, headers, max_bytes=max_bytes)

    hub = TensorDown({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    step, _r, config, _l = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=True)

    assert [model.base_model for model in step.models] == [TENSOR_ONLY]
    assert step.tensor_repos == {}
    assert [e["status"] for e in _search_log(config)["direct"]] == ["found"]


def test_a_failed_tensor_account_is_reported_with_account_and_kind_and_the_rest_goes_on():
    from modelroom.tensor_search import search_tensor_repos, tensor_search_url

    hit = {"id": "RedHatAI/Nova-7B-FP8", "tags": ["base_model:acme/Nova-7B", "base_model:quantized:acme/Nova-7B"]}
    owner_url = tensor_search_url("Nova-7B", "acme")

    def transport(method, url, headers=None, *, max_bytes=None):
        if url == owner_url:
            raise OSError("connection reset")
        return _page([hit])

    outcome = search_tensor_repos(transport, RequestBudget(10), "acme/Nova-7B")

    assert outcome.repos == ("RedHatAI/Nova-7B-FP8",)
    assert outcome.errors == ("acme: OSError",)


def test_a_programming_error_in_the_tensor_search_is_not_swallowed():
    from modelroom.tensor_search import search_tensor_repos

    def transport(method, url, headers=None, *, max_bytes=None):
        raise AttributeError("a bug")

    with pytest.raises(AttributeError):
        search_tensor_repos(transport, RequestBudget(10), "acme/Nova-7B")


# --- fix round 3 (Codex p2-b3): every expected failure of a request or an answer, per unit, never the run ----------------


def _broken_answers():
    """What the transport layer can raise for a request that did not go through, beyond `OSError` (sweep p2-b3)."""
    return [
        http.client.IncompleteRead(b"[{", 10),  # an answer cut off while its body was read
        http.client.BadStatusLine("garbage"),
        http.client.LineTooLong("header line"),
        http.client.RemoteDisconnected("closed"),
    ]


_RED_HAT_HIT = {"id": "RedHatAI/Nova-7B-FP8", "tags": ["base_model:acme/Nova-7B", "base_model:quantized:acme/Nova-7B"]}


@pytest.mark.parametrize("failure", _broken_answers(), ids=lambda exc: type(exc).__name__)
def test_a_cut_off_answer_of_one_tensor_account_keeps_the_other_accounts_hits(failure):
    owner_url = tensor_search_url("Nova-7B", "acme")

    def transport(method, url, headers=None, *, max_bytes=None):
        if url == owner_url:
            raise failure
        return _page([_RED_HAT_HIT])

    outcome = search_tensor_repos(transport, RequestBudget(10), "acme/Nova-7B")

    assert outcome.repos == ("RedHatAI/Nova-7B-FP8",)
    assert outcome.errors == (f"acme: {type(failure).__name__}",)
    assert outcome.requests == 2


@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        (Response(status=503, headers={}, body=b"busy"), "HTTP 503"),
        (Response(status=404, headers={}, body=b"[]"), "HTTP 404"),
        (Response(status=200, headers={}, body=b"not json"), "JSONDecodeError"),
        (Response(status=200, headers={}, body=b"\xff\xfe"), "UnicodeDecodeError"),
        (Response(status=200, headers={}, body=b"[" * 100_000), "RecursionError"),
        (Response(status=200, headers={}, body=b'{"error": "x"}'), "not a list"),
    ],
    ids=["503", "404", "json", "utf8", "nesting", "shape"],
)
def test_an_answer_error_of_a_tensor_account_is_reported_not_taken_for_no_offer(answer, kind):
    owner_url = tensor_search_url("Nova-7B", "acme")

    def transport(method, url, headers=None, *, max_bytes=None):
        return answer if url == owner_url else _page([_RED_HAT_HIT])

    outcome = search_tensor_repos(transport, RequestBudget(10), "acme/Nova-7B")

    assert outcome.repos == ("RedHatAI/Nova-7B-FP8",)
    assert outcome.errors == (f"acme: {kind}",)


def test_an_account_with_no_offer_is_no_error():
    outcome = search_tensor_repos(lambda *a, **k: _page([]), RequestBudget(10), "acme/Nova-7B")

    assert (outcome.repos, outcome.errors, outcome.requests) == ((), (), 2)


class _TensorCutOff(FakeHub):
    """The direct find answers; every tensor page is cut off while its body is read."""

    def __call__(self, method, url, headers=None, *, max_bytes=None):
        if "filter=safetensors" in url:
            self.calls.append(url)
            raise http.client.IncompleteRead(b"[", 10)
        return super().__call__(method, url, headers, max_bytes=max_bytes)


def test_a_cut_off_tensor_answer_keeps_the_direct_find_its_log_and_says_so_on_the_screen(tmp_path, monkeypatch):
    hub = _TensorCutOff({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    step, _r, config, lines = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=True)

    assert [model.base_model for model in step.models] == [TENSOR_ONLY]
    assert step.tensor_repos == {}
    assert [e["status"] for e in _search_log(config)["direct"]] == ["found"]
    said = [line.strip() for line in lines]
    assert f"{TENSOR_ONLY}: acme: IncompleteRead" in said
    assert f"{TENSOR_ONLY}: RedHatAI: IncompleteRead" in said


def test_without_a_vllm_machine_no_tensor_request_is_made_and_no_diagnosis_line_appears(tmp_path, monkeypatch):
    hub = _TensorCutOff({TENSOR_ONLY: _answer(TENSOR_ONLY)})

    _step1, _r, _c, lines = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=False)

    assert not any("filter=safetensors" in url for url in hub.calls)
    assert not any("IncompleteRead" in line for line in lines)


@pytest.mark.parametrize("failure", _broken_answers(), ids=lambda exc: type(exc).__name__)
def test_a_cut_off_direct_answer_is_a_model_not_read_and_the_next_one_is_still_asked(failure):
    class Broken(FakeHub):
        def __call__(self, method, url, headers=None, *, max_bytes=None):
            if "Nova-3B" in url:
                raise failure
            return super().__call__(method, url, headers, max_bytes=max_bytes)

    outcome = _direct(_catalog(), Broken({TENSOR_ONLY: _answer(TENSOR_ONLY)}), "nova")

    assert [hit.repo for hit in outcome.hits] == [TENSOR_ONLY]
    assert [(e.repo, e.status, e.reason) for e in outcome.log] == [
        (TENSOR_ONLY, "found", None),
        ("acme/Nova-3B", "not_read", f"the request failed: {type(failure).__name__}"),
    ]


@pytest.mark.parametrize("body", [b"\xff\xfe", b"[" * 100_000], ids=["utf8", "nesting"])
def test_an_unreadable_direct_answer_is_not_json(body):
    hub = FakeHub({TENSOR_ONLY: Response(status=200, headers={}, body=body)})

    [entry] = _direct(_catalog(), hub, "nova-9b").log

    assert (entry.status, entry.reason) == ("not_read", "the answer is not JSON")


@pytest.mark.parametrize("bug", [AttributeError("a bug"), KeyError("no fixture"), TypeError("a bug")], ids=lambda e: type(e).__name__)
def test_a_programming_error_in_the_direct_find_is_not_swallowed(bug):
    def transport(method, url, headers=None, *, max_bytes=None):
        raise bug

    with pytest.raises(type(bug)):
        _direct(_catalog(), transport, "nova-9b")


# --- fix round 4 (Codex p2-b4): a redirect to a URL that cannot be parsed ------------------------------------------------

_BROKEN_LOCATION = "https://[broken"


def test_a_redirect_to_a_broken_location_in_the_tensor_search_is_an_account_error_and_the_rest_goes_on():
    owner_url = tensor_search_url("Nova-7B", "acme")

    def transport(method, url, headers=None, *, max_bytes=None):
        if url == owner_url:
            return Response(status=302, headers={"Location": _BROKEN_LOCATION}, body=b"")
        return _page([_RED_HAT_HIT])

    outcome = search_tensor_repos(transport, RequestBudget(10), "acme/Nova-7B")

    assert outcome.repos == ("RedHatAI/Nova-7B-FP8",)
    assert outcome.errors == ("acme: ValueError",)


def test_a_broken_location_in_the_tensor_search_keeps_the_direct_find_its_log_and_says_so(tmp_path, monkeypatch):
    class Moved(FakeHub):
        def __call__(self, method, url, headers=None, *, max_bytes=None):
            if "filter=safetensors" in url:
                self.calls.append(url)
                return Response(status=302, headers={"Location": _BROKEN_LOCATION}, body=b"")
            return super().__call__(method, url, headers, max_bytes=max_bytes)

    step, _r, config, lines = _step(tmp_path, monkeypatch, Moved({TENSOR_ONLY: _answer(TENSOR_ONLY)}), {"search": "nova-9b"}, vllm=True)

    assert [model.base_model for model in step.models] == [TENSOR_ONLY]
    assert [e["status"] for e in _search_log(config)["direct"]] == ["found"]
    assert f"{TENSOR_ONLY}: acme: ValueError" in [line.strip() for line in lines]


def test_a_broken_location_in_the_direct_find_is_a_model_not_read_and_the_next_one_is_still_asked():
    import urllib.parse

    class Broken(FakeHub):
        """What urllib's redirect handler does with the `Location` before any redirect decision: it parses it."""

        def __call__(self, method, url, headers=None, *, max_bytes=None):
            if "Nova-3B" in url:
                urllib.parse.urlsplit(_BROKEN_LOCATION)
            return super().__call__(method, url, headers, max_bytes=max_bytes)

    outcome = _direct(_catalog(), Broken({TENSOR_ONLY: _answer(TENSOR_ONLY)}), "nova")

    assert [hit.repo for hit in outcome.hits] == [TENSOR_ONLY]
    assert [(e.repo, e.status, e.reason) for e in outcome.log] == [
        (TENSOR_ONLY, "found", None),
        ("acme/Nova-3B", "not_read", "the request failed: ValueError"),
    ]


def test_one_definition_of_the_expected_transport_failures_serves_both_searches():
    from modelroom import search_direct, search_pages, tensor_search
    from modelroom.http import BudgetExhaustedError, RedirectLimitError, ResponseTooLargeError, TransportSecurityError

    expected = search_pages.TRANSPORT_ERRORS
    for kind in (OSError, http.client.HTTPException, ValueError, TransportSecurityError, ResponseTooLargeError, RedirectLimitError):
        assert issubclass(kind, expected)
    assert not issubclass(BudgetExhaustedError, expected)
    assert search_direct.TRANSPORT_ERRORS is tensor_search.TRANSPORT_ERRORS is expected
