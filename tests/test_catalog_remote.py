"""Tests for modelroom.catalog_remote: the published catalog is loaded, cached and chosen (AP9-KA-L).

Every test goes through a `FixtureTransport` holding recorded answers in the shape of the Hub's
API (`tests/fixtures/hf_dataset_*.json`, `catalog_v2_published.json`), an injected UTC clock and a
cache folder inside `tmp_path`; nothing here opens a socket. The dataset itself does not exist
yet, so no live request is made anywhere in this suite.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import pytest

from modelroom import catalog_remote
from modelroom.catalog import CATALOG_JSON_SCHEMA_RANGE, Catalog, load_catalog
from modelroom.catalog_remote import CACHE_FILE_NAME, CatalogChoice, current_catalog
from modelroom.dialog import FileAsker
from modelroom.guided import run_guided
from modelroom.http import BudgetedTransport, FixtureTransport, RedirectingTransport, RequestBudget, Response

from fixture_support import (
    FIXTURES,
    build_transport,
    envelope_response,
    guided_transport_mapping,
    json_response,
    offline_daemon,
    offline_servers,
    windows_probes,
)

NOW = datetime(2026, 10, 3, 9, 0, 0, tzinfo=timezone.utc)
SHA = "3f5c1d9a7e" * 4  # the commit the recorded answers name
OLD_SHA = "0a1b2c3d4e" * 4
DATASET = "jm-vis/modelroom-catalog"
REVISION_URL = f"https://huggingface.co/api/datasets/{DATASET}/revision/main"
SHIPPED = load_catalog().published
assert SHIPPED is not None
KIB = 1024
DAY = timedelta(days=1)


def file_url(sha: str, name: str = "catalog-v2.json") -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{sha}/{name}"


_REDIRECT = envelope_response("hf_dataset_resolve_redirect.json")


def resolved_url(sha: str, name: str = "catalog-v2.json") -> str:
    """Where the recorded `307` leads: a relative `Location` on `/api/resolve-cache/...`, joined to the asked URL."""
    return urljoin(file_url(sha, name), _REDIRECT.header("Location").replace(SHA, sha).replace("catalog-v2.json", name))


@pytest.fixture(autouse=True)
def _published_catalog_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite's `conftest.py` pins `MODELROOM_CATALOG=shipped`; this module tests the way that is on."""
    monkeypatch.delenv("MODELROOM_CATALOG", raising=False)


def doc_text(published: date | None = None, **patch: object) -> bytes:
    """The recorded schema-2 document, with its `published` day and any top-level field changed."""
    data = json.loads((FIXTURES / "catalog_v2_published.json").read_text(encoding="utf-8"))
    data["published"] = (published or SHIPPED + 3 * DAY).isoformat()
    data.update(patch)
    return json.dumps(data, indent=2).encode("utf-8")


def hub(
    sha: str = SHA, doc: bytes | None = None, *, commit: str | None = None, name: str = "catalog-v2.json"
) -> dict[tuple[str, str], Response]:
    """The three answers of a changed dataset: revision, the `307`, then the file `name` at its resolved URL."""
    revision = json.loads((FIXTURES / "hf_dataset_revision_main.json").read_text(encoding="utf-8"))
    revision["sha"] = sha
    redirect = Response(
        status=307,
        headers={
            **_REDIRECT.headers,
            "Location": _REDIRECT.header("Location").replace(SHA, sha).replace("catalog-v2.json", name),
            "X-Repo-Commit": sha,
        },
        body=b"",
    )
    answer = Response(status=200, headers={"X-Repo-Commit": commit or sha}, body=doc if doc is not None else doc_text())
    return {
        ("GET", REVISION_URL): Response(status=200, headers={}, body=json.dumps(revision).encode("utf-8")),
        ("GET", file_url(sha, name)): redirect,
        ("GET", resolved_url(sha, name)): answer,
    }


def write_envelope(
    cache_dir: Path,
    *,
    age: timedelta,
    sha: str | None = OLD_SHA,
    doc: bytes | None = None,
    last_error: str | None = None,
    now: datetime = NOW,
) -> dict:
    checked_at = (now - age).strftime("%Y-%m-%dT%H:%M:%SZ")
    catalog = None if sha is None else json.loads(doc if doc is not None else doc_text(SHIPPED - 5 * DAY))
    envelope = {"cache_version": 1, "checked_at": checked_at, "sha": sha, "last_error": last_error, "catalog": catalog}
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / CACHE_FILE_NAME).write_text(json.dumps(envelope, indent=2), encoding="utf-8", newline="\n")
    return envelope


def read_envelope(cache_dir: Path) -> dict:
    return json.loads((cache_dir / CACHE_FILE_NAME).read_text(encoding="utf-8"))


def run(cache_dir: Path, transport, *, now: datetime = NOW, budget: RequestBudget | None = None) -> CatalogChoice:
    # the mechanism is the same on both channels: this module tests it on the old one, the new one has its own tests
    return current_catalog(
        transport, now, cache_dir, budget if budget is not None else RequestBudget(150), channel=catalog_remote.CHANNEL_V2
    )


class Offline:
    """A transport on a computer without a network: every call fails the way a socket does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, url, headers=None, *, max_bytes=None):
        self.calls.append((method, url))
        raise OSError("network is unreachable")


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".modelroom" / "catalog"


# --- the shape of the answer ---------------------------------------------------------------------


def test_the_recorded_document_is_a_valid_schema_two_catalog_dated_later_than_the_shipped_one():
    published = json.loads((FIXTURES / "catalog_v2_published.json").read_text(encoding="utf-8"))["published"]
    assert published == "2026-10-02"
    assert load_catalog(FIXTURES / "catalog_v2_published.json").model_for("acme/Nova-9B-2610") is not None


def test_a_choice_carries_the_catalog_the_screen_line_and_where_it_came_from(cache: Path):
    choice = run(cache, FixtureTransport(hub()))

    assert isinstance(choice.catalog, Catalog)
    assert choice.origin == f"catalog: published {SHIPPED + 3 * DAY}"
    assert choice.source == "published"
    assert choice.catalog.model_for("acme/Nova-9B-2610") is not None


# --- how many requests: 0, 1 or 3 -----------------------------------------------------------------


def test_a_fresh_envelope_asks_nothing(cache: Path):
    write_envelope(cache, age=timedelta(hours=1), sha=SHA, doc=doc_text())
    transport = FixtureTransport({})

    choice = run(cache, transport)

    assert transport.calls == []
    assert choice.source == "cached"
    assert choice.origin == f"catalog: published {SHIPPED + 3 * DAY}"


def test_an_old_envelope_with_the_same_sha_asks_once_and_only_renews_the_stamp(cache: Path):
    before = write_envelope(cache, age=25 * timedelta(hours=1), sha=SHA, doc=doc_text(), last_error="earlier fault")
    transport = FixtureTransport({("GET", REVISION_URL): hub(SHA)[("GET", REVISION_URL)]})

    choice = run(cache, transport)

    assert transport.calls == [("GET", REVISION_URL)]
    after = read_envelope(cache)
    assert after["checked_at"] == "2026-10-03T09:00:00Z"
    assert (after["sha"], after["catalog"]) == (before["sha"], before["catalog"])
    assert after["last_error"] is None
    assert choice.source == "cached"


def test_a_new_sha_asks_three_times_through_the_recorded_307_and_writes_one_envelope(cache: Path):
    write_envelope(cache, age=3 * DAY, sha=OLD_SHA)
    transport = FixtureTransport(hub())

    choice = run(cache, transport)

    assert transport.calls == [("GET", REVISION_URL), ("GET", file_url(SHA)), ("GET", resolved_url(SHA))]
    assert resolved_url(SHA).startswith("https://huggingface.co/api/resolve-cache/datasets/")
    envelope = read_envelope(cache)
    assert envelope["sha"] == SHA and envelope["last_error"] is None
    assert envelope["catalog"] == json.loads(doc_text())
    assert envelope["cache_version"] == 1 and envelope["checked_at"] == "2026-10-03T09:00:00Z"
    assert choice.source == "published"


def test_the_first_run_without_any_envelope_asks_three_times_too(cache: Path):
    transport = FixtureTransport(hub())

    run(cache, transport)

    assert len(transport.calls) == 3
    assert read_envelope(cache)["sha"] == SHA


def test_the_revision_url_and_the_file_url_are_built_from_the_dataset_constant_alone():
    assert catalog_remote.DATASET == DATASET
    assert catalog_remote.REMOTE_FILE_NAME == "catalog-v2.json"
    assert catalog_remote.CACHE_FILE_NAME == "catalog-v2.cache.json"


# --- the switch ------------------------------------------------------------------------------------


def test_modelroom_catalog_shipped_asks_nothing_and_writes_no_envelope(cache: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODELROOM_CATALOG", "shipped")
    transport = FixtureTransport(hub())

    choice = run(cache, transport)

    assert transport.calls == []
    assert not cache.exists()
    assert choice.source == "shipped"
    assert choice.origin == f"catalog: shipped {SHIPPED}"


# --- the shared request budget -------------------------------------------------------------------


def test_a_budget_of_none_is_no_attempt_and_books_nothing(cache: Path):
    transport = FixtureTransport(hub())
    budget = RequestBudget(0)

    choice = run(cache, transport, budget=budget)

    assert transport.calls == [] and budget.used == 0
    assert not cache.exists()  # an attempt that never sent a request is not the day's attempt
    assert choice.source == "shipped"
    assert "budget exhausted" in choice.origin


def test_a_budget_that_ends_after_the_revision_costs_one_request_and_books_the_attempt(cache: Path):
    transport = FixtureTransport(hub())
    budget = RequestBudget(1)

    choice = run(cache, transport, budget=budget)

    assert transport.calls == [("GET", REVISION_URL)] and budget.used == 1
    assert read_envelope(cache)["last_error"] == "budget exhausted"
    assert read_envelope(cache)["catalog"] is None
    assert choice.source == "shipped"


def test_a_budget_that_ends_after_the_307_costs_two_requests_and_books_the_attempt(cache: Path):
    transport = FixtureTransport(hub())
    budget = RequestBudget(2)

    choice = run(cache, transport, budget=budget)

    assert transport.calls == [("GET", REVISION_URL), ("GET", file_url(SHA))] and budget.used == 2
    assert "budget exhausted" in read_envelope(cache)["last_error"]
    assert choice.source == "shipped"
    # booked: a second run an hour later asks nothing
    again = FixtureTransport(hub())
    run(cache, again, now=NOW + timedelta(hours=1))
    assert again.calls == []


def test_the_budget_is_the_shared_one_of_the_guided_run(cache: Path):
    budget = RequestBudget(150)

    run(cache, FixtureTransport(hub()), budget=budget)

    assert budget.used == 3


# --- the hard hop limit: 3 requests, whatever MAX_REDIRECTS says ---------------------------------


def test_a_revision_that_answers_307_is_not_followed(cache: Path):
    mapping = hub()
    mapping[("GET", REVISION_URL)] = Response(status=307, headers={"Location": "/api/elsewhere"}, body=b"")
    transport = FixtureTransport(mapping)

    choice = run(cache, transport)

    assert transport.calls == [("GET", REVISION_URL)]
    assert choice.source == "shipped" and "redirects" in choice.origin


def test_a_file_that_redirects_twice_is_stopped_after_exactly_three_requests(cache: Path):
    mapping = hub()
    second = resolved_url(SHA)
    mapping[("GET", second)] = Response(status=307, headers={"Location": "/api/next", "X-Repo-Commit": SHA}, body=b"")
    transport = FixtureTransport(mapping)

    choice = run(cache, transport)

    assert transport.calls == [("GET", REVISION_URL), ("GET", file_url(SHA)), ("GET", second)]
    assert choice.source == "shipped" and "redirects" in choice.origin


def test_five_redirects_in_a_row_still_stop_at_three_calls_on_the_transport(cache: Path):
    mapping = hub()
    mapping[("GET", file_url(SHA))] = Response(status=307, headers={"Location": "/r1"}, body=b"")
    for number in range(1, 6):
        mapping[("GET", f"https://huggingface.co/r{number}")] = Response(
            status=307, headers={"Location": f"/r{number + 1}"}, body=b""
        )
    transport = FixtureTransport(mapping)

    run(cache, transport)

    assert len(transport.calls) == 3


def test_a_redirect_to_a_foreign_host_is_a_failure_and_is_never_asked(cache: Path):
    mapping = hub()
    mapping[("GET", file_url(SHA))] = Response(status=307, headers={"Location": "https://evil.example/catalog-v2.json"}, body=b"")
    transport = FixtureTransport(mapping)
    budget = RequestBudget(150)

    choice = run(cache, transport, budget=budget)

    assert ("GET", "https://evil.example/catalog-v2.json") not in transport.calls
    assert len(transport.calls) == 2 and budget.used == 3  # the refused hop was booked, never sent
    assert choice.source == "shipped" and read_envelope(cache)["last_error"]


def test_other_callers_of_the_redirecting_transport_keep_following_up_to_five_hops():
    mapping = {("GET", f"https://example.test/{number}"): Response(
        status=302, headers={"Location": f"https://example.test/{number + 1}"}, body=b"") for number in range(1, 5)}
    mapping[("GET", "https://example.test/5")] = Response(status=200, headers={}, body=b"ok")
    inner = FixtureTransport(mapping)

    assert RedirectingTransport(inner)("GET", "https://example.test/1").status == 200
    assert len(inner.calls) == 5


# --- the longest answer, through every layer ------------------------------------------------------


def test_each_hop_is_handed_the_limit_of_its_call(cache: Path):
    transport = FixtureTransport(hub())

    run(cache, transport)

    assert transport.limits == [64 * KIB, 1024 * KIB, 1024 * KIB]  # the 307 hop inherits the file's limit


def test_a_revision_answer_one_byte_over_64_kib_is_refused(cache: Path):
    mapping = hub()
    mapping[("GET", REVISION_URL)] = Response(status=200, headers={}, body=b" " * (64 * KIB + 1))

    choice = run(cache, FixtureTransport(mapping))

    assert choice.source == "shipped" and f"longer than {64 * KIB} bytes" in choice.origin


def test_a_catalog_one_byte_over_1_mib_is_refused(cache: Path):
    mapping = hub(doc=b" " * (1024 * KIB + 1))

    choice = run(cache, FixtureTransport(mapping))

    assert choice.source == "shipped" and f"longer than {1024 * KIB} bytes" in choice.origin
    assert read_envelope(cache)["catalog"] is None


def test_an_oversized_error_answer_is_refused_as_too_large_not_read_as_a_404(cache: Path):
    mapping = hub()
    mapping[("GET", REVISION_URL)] = Response(status=404, headers={}, body=b"x" * (64 * KIB + 1))

    choice = run(cache, FixtureTransport(mapping))

    assert "longer than" in choice.origin and "404" not in choice.origin


def test_a_body_shorter_than_announced_is_a_failure(cache: Path):
    mapping = hub()
    body = doc_text()
    mapping[("GET", resolved_url(SHA))] = Response(
        status=200, headers={"X-Repo-Commit": SHA, "Content-Length": str(len(body) + 100)}, body=body
    )

    choice = run(cache, FixtureTransport(mapping))

    assert choice.source == "shipped" and "ended early" in choice.origin


def test_the_limit_travels_through_the_budgeted_and_the_redirecting_layer_to_the_307_target():
    inner = FixtureTransport(hub())
    layered = RedirectingTransport(BudgetedTransport(inner, RequestBudget(10)), max_hops=2)

    layered("GET", file_url(SHA), max_bytes=500_000)

    assert inner.limits == [500_000, 500_000]


# --- the clock ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "asked"),
    [
        (timedelta(hours=23, minutes=59, seconds=59), 0),
        (timedelta(hours=24), 1),
        (timedelta(hours=24, seconds=1), 1),
        (timedelta(seconds=-1), 1),  # `checked_at` in the future: the clock was set back
        (timedelta(days=-30), 1),
        (timedelta(0), 0),
    ],
)
def test_the_age_of_the_envelope_decides_whether_one_attempt_is_made(cache: Path, age: timedelta, asked: int):
    write_envelope(cache, age=age, sha=SHA, doc=doc_text())
    transport = FixtureTransport({("GET", REVISION_URL): hub()[("GET", REVISION_URL)]})

    run(cache, transport)

    assert len(transport.calls) == asked


def test_a_run_that_failed_is_not_repeated_the_same_day_not_even_by_a_second_run(cache: Path):
    first = Offline()
    run(cache, first)
    second, third = Offline(), Offline()

    run(cache, second, now=NOW + timedelta(hours=1))
    run(cache, third, now=NOW + timedelta(hours=23, minutes=59, seconds=59))
    fourth = Offline()
    run(cache, fourth, now=NOW + timedelta(hours=24))

    assert len(first.calls) == 1  # the revision, which never answered
    assert second.calls == [] and third.calls == []
    assert len(fourth.calls) == 1
    assert "unreachable" in read_envelope(cache)["last_error"]


def test_a_naive_clock_is_read_as_utc(cache: Path):
    write_envelope(cache, age=timedelta(hours=1), sha=SHA, doc=doc_text())
    transport = FixtureTransport({})

    run(cache, transport, now=NOW.replace(tzinfo=None))

    assert transport.calls == []


# --- the envelope: what counts as one ------------------------------------------------------------


def _bad_envelopes() -> dict[str, str]:
    good = {"cache_version": 1, "checked_at": "2026-10-03T08:00:00Z", "sha": SHA, "last_error": None,
            "catalog": json.loads(doc_text())}
    broken_catalog = {**good, "catalog": {**good["catalog"], "unknown_field": 1}}
    return {
        "not_json": "{ this is not json",
        "not_an_object": "[1, 2]",
        "wrong_cache_version": json.dumps({**good, "cache_version": 2}),
        "catalog_invalid": json.dumps(broken_catalog),
        "sha_not_hex": json.dumps({**good, "sha": "z" * 40}),
        "sha_too_short": json.dumps({**good, "sha": "abc123"}),
        "sha_without_catalog": json.dumps({**good, "catalog": None}),
        "catalog_without_sha": json.dumps({**good, "sha": None}),
        "missing_key": json.dumps({key: value for key, value in good.items() if key != "last_error"}),
        "extra_key": json.dumps({**good, "extra": 1}),
        "stamp_without_zone": json.dumps({**good, "checked_at": "2026-10-03T08:00:00"}),
        "stamp_not_a_date": json.dumps({**good, "checked_at": "yesterday"}),
    }


@pytest.mark.parametrize("name", sorted(_bad_envelopes()))
def test_an_envelope_that_does_not_read_or_validate_counts_as_absent(cache: Path, name: str):
    cache.mkdir(parents=True)
    (cache / CACHE_FILE_NAME).write_text(_bad_envelopes()[name], encoding="utf-8")
    transport = FixtureTransport(hub())

    choice = run(cache, transport)  # a fresh-looking stamp in a bad envelope must not suppress the attempt

    assert len(transport.calls) == 3
    assert choice.source == "published"
    assert read_envelope(cache)["sha"] == SHA


def test_a_temporary_file_left_by_an_interrupted_write_changes_nothing(cache: Path):
    write_envelope(cache, age=timedelta(hours=1), sha=SHA, doc=doc_text())
    (cache / f"{CACHE_FILE_NAME}.4242.tmp").write_text('{"cache_version": 1, "chec', encoding="utf-8")
    transport = FixtureTransport({})

    choice = run(cache, transport)

    assert transport.calls == [] and choice.source == "cached"
    # and the next write replaces the envelope as a whole, leaving the stray file where it was
    run(cache, FixtureTransport(hub()), now=NOW + 2 * DAY)
    assert read_envelope(cache)["sha"] == SHA
    assert (cache / f"{CACHE_FILE_NAME}.4242.tmp").exists()


def test_a_stamp_that_overflows_in_utc_counts_as_absent_and_the_run_goes_on(cache: Path):
    write_envelope(cache, age=timedelta(hours=1), sha=SHA, doc=doc_text())
    path = cache / CACHE_FILE_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["checked_at"] = "0001-01-01T00:00:00+01:00"  # passes `fromisoformat`, overflows at `astimezone(utc)`
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    transport = FixtureTransport(hub())

    choice = run(cache, transport)

    assert len(transport.calls) == 3
    assert choice.source == "published"
    assert read_envelope(cache)["checked_at"] == "2026-10-03T09:00:00Z"


def test_an_envelope_longer_than_the_bound_counts_as_absent_even_when_its_start_is_valid(cache: Path):
    write_envelope(cache, age=timedelta(hours=1), sha=SHA, doc=doc_text())
    path = cache / CACHE_FILE_NAME
    bound = catalog_remote.CHANNEL_V2.envelope_max_bytes  # 2 MiB + 4 KiB for the schema-2 channel
    path.write_bytes(path.read_bytes() + b" " * (bound + 1))
    transport = FixtureTransport(hub())

    run(cache, transport)

    assert len(transport.calls) == 3
    assert path.stat().st_size < bound  # replaced whole by the new envelope


def test_two_runs_one_after_the_other_replace_the_envelope_whole_never_a_mixture(cache: Path):
    # Runs that overlap are covered by `atomic_write_json` (one `os.replace`); rule 2 asks for no lock,
    # so this test does not pretend to race two writers.
    other_sha = "7d2e9b4a6c" * 4
    first = hub(SHA, doc_text(SHIPPED + 3 * DAY))
    second = hub(other_sha, doc_text(SHIPPED + 4 * DAY))

    run(cache, FixtureTransport(first), now=NOW)
    after_first = read_envelope(cache)
    run(cache, FixtureTransport(second), now=NOW + 2 * DAY)
    after_second = read_envelope(cache)

    assert (after_first["sha"], after_first["catalog"]["published"]) == (SHA, (SHIPPED + 3 * DAY).isoformat())
    assert (after_second["sha"], after_second["catalog"]["published"]) == (other_sha, (SHIPPED + 4 * DAY).isoformat())
    assert sorted(path.name for path in cache.iterdir()) == [CACHE_FILE_NAME]  # no temporary file stays


# --- one rule of choice, for success and failure alike ---------------------------------------------


def _chosen(cache: Path, *, envelope: int | None, fresh: int | None, offline: bool = False) -> CatalogChoice:
    """Days of `published` relative to the shipped catalog: the envelope's, the freshly loaded one's."""
    if envelope is not None:
        write_envelope(cache, age=3 * DAY, sha=OLD_SHA, doc=doc_text(SHIPPED + envelope * DAY))
    if offline:
        return run(cache, Offline())
    return run(cache, FixtureTransport(hub(SHA, doc_text(SHIPPED + fresh * DAY))))


def test_a_shipped_catalog_younger_than_the_envelope_wins(cache: Path):
    choice = _chosen(cache, envelope=-4, fresh=None, offline=True)

    assert choice.source == "shipped"
    assert choice.origin.startswith(f"catalog: shipped {SHIPPED} (the published one did not read")


def test_a_tie_of_the_envelope_and_the_shipped_catalog_goes_to_the_envelope(cache: Path):
    choice = _chosen(cache, envelope=0, fresh=None, offline=True)

    assert choice.source == "cached"


def test_a_freshly_loaded_catalog_as_old_as_the_shipped_one_wins_the_tie(cache: Path):
    choice = _chosen(cache, envelope=None, fresh=0)

    assert choice.source == "published"
    assert choice.origin == f"catalog: published {SHIPPED}"


def test_a_freshly_loaded_catalog_as_old_as_the_envelope_one_is_the_one_chosen(cache: Path):
    choice = _chosen(cache, envelope=2, fresh=2)

    assert choice.source == "published"


def test_an_envelope_newer_than_the_download_is_chosen_and_the_download_is_still_stored(cache: Path):
    write_envelope(cache, age=3 * DAY, sha=OLD_SHA, doc=doc_text(SHIPPED + 5 * DAY), last_error="earlier fault")

    choice = run(cache, FixtureTransport(hub(SHA, doc_text(SHIPPED + 3 * DAY))))

    assert choice.source == "cached"
    assert choice.origin == f"catalog: published {SHIPPED + 5 * DAY}"  # the old fault is hidden: this attempt worked
    envelope = read_envelope(cache)
    assert envelope["sha"] == SHA and envelope["last_error"] is None
    assert envelope["catalog"] == json.loads(doc_text(SHIPPED + 3 * DAY))


def test_a_download_newer_than_the_envelope_and_the_shipped_one_is_chosen(cache: Path):
    write_envelope(cache, age=3 * DAY, sha=OLD_SHA, doc=doc_text(SHIPPED + 3 * DAY))

    choice = run(cache, FixtureTransport(hub(SHA, doc_text(SHIPPED + 5 * DAY))))

    assert choice.source == "published"
    assert choice.origin == f"catalog: published {SHIPPED + 5 * DAY}"


def test_a_download_as_old_as_the_envelope_wins_the_tie_and_the_old_fault_is_not_shown(cache: Path):
    write_envelope(cache, age=3 * DAY, sha=OLD_SHA, doc=doc_text(SHIPPED + 3 * DAY), last_error="earlier fault")

    choice = run(cache, FixtureTransport(hub(SHA, doc_text(SHIPPED + 3 * DAY))))

    assert choice.source == "published"
    assert choice.origin == f"catalog: published {SHIPPED + 3 * DAY}"


def test_a_freshly_loaded_catalog_older_than_the_shipped_one_is_stored_but_not_chosen(cache: Path):
    choice = _chosen(cache, envelope=None, fresh=-2)

    assert choice.source == "shipped" and choice.origin == f"catalog: shipped {SHIPPED}"
    assert read_envelope(cache)["catalog"]["published"] == (SHIPPED - 2 * DAY).isoformat()


def test_offline_with_an_envelope_older_than_the_shipped_one_the_shipped_one_is_chosen(cache: Path):
    assert _chosen(cache, envelope=-1, fresh=None, offline=True).source == "shipped"


def test_offline_without_an_envelope_the_shipped_one_is_chosen(cache: Path):
    assert _chosen(cache, envelope=None, fresh=None, offline=True).source == "shipped"


def test_offline_with_a_newer_envelope_the_envelope_is_chosen(cache: Path):
    choice = _chosen(cache, envelope=3, fresh=None, offline=True)

    assert choice.source == "cached"
    assert choice.origin.startswith(f"catalog: published {SHIPPED + 3 * DAY} (the published one did not read")


# --- a broken dataset ends in the same rule, never in an abort -----------------------------------


def _answer(mapping: dict, key: str, response: Response) -> None:
    mapping[("GET", {"revision": REVISION_URL, "file": file_url(SHA), "target": resolved_url(SHA)}[key])] = response


def _doc_with(**patch: object) -> bytes:
    return doc_text(**patch)


_BROKEN = {
    "revision_404": lambda m: _answer(m, "revision", Response(404, {}, b'{"error": "Repository not found"}')),
    "revision_500": lambda m: _answer(m, "revision", Response(500, {}, b"upstream error")),
    "revision_not_json": lambda m: _answer(m, "revision", Response(200, {}, b"<html>maintenance</html>")),
    "revision_without_sha": lambda m: _answer(m, "revision", Response(200, {}, b'{"id": "x"}')),
    "revision_sha_not_hex": lambda m: _answer(m, "revision", Response(200, {}, b'{"sha": "' + b"g" * 40 + b'"}')),
    "revision_is_a_list": lambda m: _answer(m, "revision", Response(200, {}, b"[1]")),
    "file_404": lambda m: _answer(m, "target", Response(404, {"X-Repo-Commit": SHA}, b"missing")),
    "file_500": lambda m: _answer(m, "target", Response(500, {"X-Repo-Commit": SHA}, b"boom")),
    "file_not_json": lambda m: _answer(m, "target", Response(200, {"X-Repo-Commit": SHA}, b"not json")),
    "file_not_utf8": lambda m: _answer(m, "target", Response(200, {"X-Repo-Commit": SHA}, b"\xff\xfe{}")),
    "file_without_schema_version": lambda m: _answer(m, "target", Response(200, {"X-Repo-Commit": SHA}, b'{"models": []}')),
    "file_wrong_schema": lambda m: _answer(m, "target", Response(200, {"X-Repo-Commit": SHA}, _doc_with(surprise=1))),
    "file_limit_exceeded": lambda m: _answer(m, "target", Response(200, {"X-Repo-Commit": SHA}, _doc_with(
        publishers=[{"account": "acme", "name": "n" * 301, "region": None}]))),
    "file_is_schema_three": lambda m: _answer(m, "target", Response(200, {"X-Repo-Commit": SHA}, _doc_with(schema_version=3))),
    "commit_header_differs": lambda m: _answer(m, "target", Response(200, {"X-Repo-Commit": "9" * 40}, doc_text())),
    "commit_header_missing": lambda m: _answer(m, "target", Response(200, {}, doc_text())),
    "foreign_host": lambda m: _answer(m, "file", Response(307, {"Location": "https://evil.example/x"}, b"")),
    "second_redirect": lambda m: _answer(m, "target", Response(307, {"Location": "/again"}, b"")),
}


@pytest.mark.parametrize("name", sorted(_BROKEN))
def test_a_broken_dataset_falls_back_keeps_the_envelope_catalog_and_records_the_reason(cache: Path, name: str):
    before = write_envelope(cache, age=3 * DAY, sha=OLD_SHA, doc=doc_text(SHIPPED - 5 * DAY))
    mapping = hub()
    _BROKEN[name](mapping)

    choice = run(cache, FixtureTransport(mapping))

    assert choice.source == "shipped"
    assert choice.origin.startswith(f"catalog: shipped {SHIPPED} (the published one did not read: ")
    reason = choice.origin.split("did not read: ", 1)[1][:-1]
    assert 0 < len(reason) <= 97 and reason.isascii() and reason.isprintable()
    after = read_envelope(cache)
    assert (after["sha"], after["catalog"]) == (before["sha"], before["catalog"])
    assert after["checked_at"] == "2026-10-03T09:00:00Z" and after["last_error"]


def test_a_dataset_that_says_schema_three_gives_a_note_and_the_fallback(cache: Path):
    mapping = hub(doc=doc_text(schema_version=3))

    choice = run(cache, FixtureTransport(mapping))

    assert choice.source == "shipped" and "schema_version 3" in choice.origin


def test_the_reason_of_a_failure_is_one_clean_ascii_line_cut_to_97_characters(cache: Path):
    mapping = hub(doc=doc_text(surprise="\x1b[2Jé" + "x" * 400))

    choice = run(cache, FixtureTransport(mapping))

    assert "\x1b" not in choice.origin and choice.origin.isascii()
    assert len(choice.origin.split("did not read: ", 1)[1][:-1]) <= 97
    assert "\x1b" not in read_envelope(cache)["last_error"]


def test_a_reason_stored_by_an_earlier_attempt_is_named_with_its_date(cache: Path):
    write_envelope(cache, age=timedelta(hours=1), sha=SHA, doc=doc_text(SHIPPED - 5 * DAY), last_error="HTTP 404")

    choice = run(cache, FixtureTransport({}))

    assert choice.origin == f"catalog: shipped {SHIPPED} (the published one did not read on 2026-10-03: HTTP 404)"


def test_a_stored_reason_with_control_characters_is_cleaned_before_it_is_shown(cache: Path):
    write_envelope(cache, age=timedelta(hours=1), sha=SHA, doc=doc_text(SHIPPED - 5 * DAY), last_error="\x1b[31mred\nline")

    choice = run(cache, FixtureTransport({}))

    assert "\x1b" not in choice.origin and "\n" not in choice.origin


def test_a_cache_folder_that_cannot_be_written_asks_again_on_the_next_run_but_chooses_the_same(tmp_path: Path):
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("a file where the cache folder should be", encoding="utf-8")
    first, second = FixtureTransport(hub()), FixtureTransport(hub())

    one = run(blocked, first)
    two = run(blocked, second)  # the same `now`: no stamp could be kept, so the day's attempt is asked again

    assert one.source == two.source == "published" and one.origin == two.origin
    assert len(first.calls) == 3 and len(second.calls) == 3
    assert blocked.read_text(encoding="utf-8") == "a file where the cache folder should be"


def test_a_runtime_error_of_the_transport_is_reported_with_its_own_text_not_as_a_redirect(cache: Path):
    class Vanishing:
        def __call__(self, method, url, headers=None, *, max_bytes=None):
            raise RuntimeError("the socket vanished")

    choice = run(cache, Vanishing())

    assert choice.source == "shipped"
    assert "the socket vanished" in choice.origin and "redirects" not in choice.origin


# --- who may load it ---------------------------------------------------------------------------------


def test_only_the_guided_run_imports_the_loader_and_only_readers_of_schema_two_fetch_the_file():
    package = Path(catalog_remote.__file__).parent
    users = sorted(
        path.name
        for path in package.glob("*.py")
        if re.search(r"^\s*(from \.catalog_remote import|from \. import .*catalog_remote|import .*catalog_remote)",
                     path.read_text(encoding="utf-8"), re.MULTILINE)
    )
    # `cli.py` is the second user since AP9-KO, for `offline_catalog` alone: it reads the stored file and asks no one.
    # The loader that asks (`current_catalog`) stays with the guided run (checked below).
    assert users == ["cli.py", "guided.py"]  # 0.1.0 ships no such module at all; that is a fact of the release, not something this test proves
    askers = sorted(
        path.name for path in package.glob("*.py") if "current_catalog" in path.read_text(encoding="utf-8") and path.name != "catalog_remote.py"
    )
    assert askers == ["guided.py"]
    low, high = CATALOG_JSON_SCHEMA_RANGE
    assert low <= 2 < high  # the file the 0.2.0 reader asks for is one this reader accepts
    assert catalog_remote.REMOTE_FILE_NAME == "catalog-v2.json" and catalog_remote.CACHE_FILE_NAME.startswith("catalog-v2.")


# --- the guided run ------------------------------------------------------------------------------------


def test_a_guided_run_with_no_word_asks_the_current_models_of_the_published_catalog(tmp_path: Path):
    from modelroom.search_pages import catalog_page_url

    new_file = json.loads(doc_text())
    new_file["schema_version"] = 3  # a guided run asks the new channel
    mapping = {**guided_transport_mapping(), **hub(doc=json.dumps(new_file).encode("utf-8"), name="catalog-v3.json")}
    empty = json_response("hf_search_none.json")
    published = load_catalog(FIXTURES / "catalog_v2_published.json")
    # only the published catalog's current models are bound: a page asked for any other catalog is a KeyError
    for model in (m for family in published.families for m in family.models if m.latest and m.kind == "text"):
        mapping[("GET", catalog_page_url(model.hf_repo.partition("/")[2]))] = empty
    transport = build_transport(mapping)
    lines: list[str] = []

    run_guided(
        FileAsker({"results": "here", "machines": ["this-machine"], "machine_class": "laptop", "search": "",
                   "filter_owners": True, "select": [], "users": 1, "context": "8192"}),
        here=tmp_path / "results",
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=transport,
        probes=windows_probes(),
        daemon=offline_daemon(),
        servers=offline_servers(),
        now=NOW,
        out=lines.append,
    )

    asked = [url for _method, url in transport.calls]
    assert catalog_page_url("Nova-9B-2610") in asked  # a model only the published catalog knows is a page
    assert any(line.strip() == f"catalog: published {SHIPPED + 3 * DAY}" for line in lines)
    assert (tmp_path / "home" / ".modelroom" / "catalog" / catalog_remote.CHANNEL_V3.cache_name).exists()
