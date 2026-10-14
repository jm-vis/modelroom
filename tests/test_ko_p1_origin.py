"""AP9-KO P1, part 2: the origin of a package steers only the automatic recommendation (V4, stage 2 of E1).

A GGUF package of an account that neither owns the base model nor is a listed packager (`origin` `other` on its row)
is never the automatic choice -- whatever its check, approval or measurement says. It stays a choice by name; only an
`absent` check blocks that. The recommendation looks at the rows shown, passes over what it may not recommend with
the line `COMMUNITY_CONVERSION`, and where nothing is left says `NO_RECOMMENDATION` and pulls nothing: no implicit
pull, and an answer file without `pull_package` does not move to another rank. Ranking, fit and rating stay as they are.
"""

from __future__ import annotations

from pathlib import Path

from modelroom.catalog_findings import NO_RECOMMENDATION, PackageCheck, community_conversion_text
from modelroom.daemon import PULL_PATH
from modelroom.guided_install import PullContext, pull_choices, pull_step
from modelroom.pull_choice import ambiguous_names, default_row, offered, pull_rows
from modelroom.screen import plain
from modelroom.views import RunView, terminal_blocks

from fixture_rows import document, ollama_package, package
from fixture_servers import pull_daemon, tags_with
from test_ko_render import _build, _catalog_listing
from test_pull_choice import _Recording, _daemon, _run
from test_render import GIB, _hf_package

LISTED = "unsloth/Nova-7B-GGUF"  # the account does not matter to the rule: the document's `origin` decides
FOREIGN = "somebody/Nova-7B-GGUF"
Q8 = package(LISTED, "Nova-7B-Q8_0.gguf", "Q8_0")
Q5 = package(LISTED, "Nova-7B-Q5_K_M.gguf", "Q5_K_M")
Q4 = package(LISTED, "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
F8 = package(FOREIGN, "Nova-7B-Q8_0.gguf", "Q8_0")
F4 = package(FOREIGN, "Nova-7B-Q4_K_M.gguf", "Q4_K_M")


def _marked(doc, *ranks: int, **extra):
    """`doc` with the rows at `ranks` marked as the document of the render marks a foreign GGUF (`origin` `other`)."""
    block = doc.machines[0]
    ranked = [e.model_copy(update={"origin": "other", **extra}) if e.rank in ranks else e for e in block.ranked]
    return doc.model_copy(update={"machines": [block.model_copy(update={"ranked": ranked})]})


def _text(doc, packages) -> str:
    view = RunView(install_for="workstation", packages=packages)
    return "\n".join(plain(line) for line in terminal_blocks(doc, view=view))


def _install_line(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("install") or "ollama pull" in line]


# --- the install line ----------------------------------------------------------------------------------------------


def test_a_foreign_conversion_at_the_top_is_passed_over_with_the_community_line_and_the_next_row_is_offered():
    doc = _marked(document([F8, Q5]), 1)

    text = _text(doc, [F8, Q5])

    assert community_conversion_text("somebody") in text
    assert "ollama pull hf.co/unsloth/Nova-7B-GGUF:Q5_K_M" in text
    assert "ollama pull hf.co/somebody/Nova-7B-GGUF:Q8_0" not in text


def test_a_present_check_an_approval_or_a_measurement_do_not_make_a_foreign_conversion_the_choice():
    """Independent of `present`, of the provenance and of a measurement (V4)."""
    present = PackageCheck.model_validate(
        {
            "repo": FOREIGN, "revision": F8.revision or "a" * 40, "identity": "Nova-7B-Q8_0.gguf", "file": "Nova-7B-Q8_0.gguf",
            "quantization": "Q8_0", "architecture": "nova7", "architecture_source": "file_header", "result": "present",
            "llamacpp_commit": "45451c366d4256f94fbc8845645e31a6b9155054",
            "source_url": "https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/src/llama-arch.cpp",
            "checked": "2026-10-13",
        }
    )
    approved = F8.model_copy(update={"provenance": "approved"})
    doc = _marked(document([approved, Q5]), 1, load_check=present, measurement_group=0, measurement_id="20260923T083000Z-5c1e9a07", speed_tps=50.4)

    text = _text(doc, [approved, Q5])

    assert "ollama pull hf.co/unsloth/Nova-7B-GGUF:Q5_K_M" in text and "somebody/Nova-7B-GGUF:Q8_0" not in "\n".join(_install_line(text))
    assert default_row(pull_rows(doc, "workstation")) is None


def test_where_every_row_shown_is_foreign_nothing_is_offered_and_the_line_says_so():
    doc = _marked(document([F8, F4]), 1, 2)

    text = _text(doc, [F8, F4])

    assert "ollama pull" not in text
    assert community_conversion_text("somebody") in text and text.count(community_conversion_text("somebody")) == 1
    assert NO_RECOMMENDATION in text


def test_only_the_rows_shown_are_looked_at_a_hidden_alternative_is_neither_shown_nor_searched():
    """The best row is a foreign GGUF and the allowed alternative lies outside the display: nothing is recommended."""
    doc = _marked(document([F8], total=5), 1)

    text = _text(doc, [F8, Q4])  # `Q4` is in the run's packages but in no shown row

    assert "ollama pull" not in text and NO_RECOMMENDATION in text


def test_a_registry_package_and_an_own_conversion_are_recommended():
    registry = ollama_package("granite4.2:8b")
    doc = document([registry, Q5])

    assert all(row.community is None and row.recommendable for row in pull_rows(doc, "workstation"))
    assert "ollama pull granite4.2:8b" in _text(doc, [registry, Q5])


# --- the pull of step 5 ---------------------------------------------------------------------------------------------


def test_an_answer_file_without_pull_package_pulls_nothing_and_does_not_move_to_rank_two(tmp_path: Path):
    doc = _marked(document([F8, Q5]), 1)
    context = PullContext(doc, "workstation", [F8, Q5], ambiguous_names([F8, Q5]))
    daemon = _daemon()
    run, lines = _run(tmp_path, _Recording({"pull": True}), daemon)

    pull_step(run, context, True)

    assert not any(call[1] == PULL_PATH for call in daemon.calls)
    text = "\n".join(lines)
    # rank 2 is recommendable (and the install line offered it): the line says why rank 1 is passed over, but not
    # that every row is blocked or unchecked -- and still no rank change
    assert community_conversion_text("somebody") in text and NO_RECOMMENDATION not in text
    assert default_row(pull_rows(doc, "workstation")) is None  # rank 1 or nothing


def test_without_pull_package_and_with_every_row_foreign_both_lines_are_said(tmp_path: Path):
    doc = _marked(document([F8, F4]), 1, 2)
    context = PullContext(doc, "workstation", [F8, F4], ambiguous_names([F8, F4]))
    run, lines = _run(tmp_path, _Recording({"pull": True}), _daemon())

    pull_step(run, context, True)

    text = "\n".join(lines)
    assert community_conversion_text("somebody") in text and NO_RECOMMENDATION in text


def test_an_explicit_choice_of_a_foreign_conversion_is_still_possible(tmp_path: Path):
    name = "hf.co/somebody/Nova-7B-GGUF:Q8_0"
    doc = _marked(document([F8, Q5]), 1)
    context = PullContext(doc, "workstation", [F8, Q5], ambiguous_names([F8, Q5]))
    daemon = pull_daemon(after_pull=tags_with(name))
    run, _lines = _run(tmp_path, _Recording({"pull_package": name, "pull": True}), daemon)

    pull_step(run, context, True)

    assert any(call[1] == PULL_PATH for call in daemon.calls)


def test_the_pointer_of_the_list_never_starts_on_a_foreign_conversion():
    doc = _marked(document([F8, Q5]), 1)
    choices = pull_choices(pull_rows(doc, "workstation"), set())
    assert [choice.checked for choice in choices] == [False, True, False]  # starts on #2

    every = _marked(document([F8, F4]), 1, 2)
    choices = pull_choices(pull_rows(every, "workstation"), set())
    assert [(choice.value, choice.checked) for choice in choices if choice.checked] == [("none", True)]

    plain_doc = document([Q8, Q5])  # nothing foreign: nothing marked, the pointer starts where it always did
    assert not any(choice.checked for choice in pull_choices(pull_rows(plain_doc, "workstation"), set()))


# --- the rule itself -------------------------------------------------------------------------------------------------


def test_offered_passes_a_rank_none_of_whose_rows_is_recommendable_and_default_row_never_moves_on():
    rows = pull_rows(_marked(document([F8, Q5]), 1), "workstation")

    assert offered(rows).row.rank == 2 and [row.rank for row in offered(rows).passed] == [1]
    assert offered(rows, [1], past_disk=False).row is None  # an answer file without `pull_package`: rank 1 or nothing
    assert default_row(rows) is None
    assert offered(pull_rows(_marked(document([F8, F4]), 1, 2), "workstation")).row is None


# --- from the render: who is `other` ---------------------------------------------------------------------------------


def test_the_publisher_a_listed_packager_and_a_stranger_through_the_render():
    own = _hf_package(repo="acme/Nova-8B-GGUF")
    listed = _hf_package(repo="packager/Nova-8B-GGUF", quantization="Q8_0", weights_bytes=8 * GIB)
    stranger = _hf_package(repo="stranger/Nova-8B-GGUF", quantization="Q6_K", weights_bytes=7 * GIB)
    packages = [own, listed, stranger]
    doc = _build(packages, _catalog_listing(["packager"]))

    by_packager = {row.cell.split(" · ")[0]: row for row in pull_rows(doc, "workstation", packages=packages)}

    assert by_packager["acme"].recommendable and by_packager["packager"].recommendable
    assert by_packager["stranger"].community == "stranger" and not by_packager["stranger"].recommendable
    assert by_packager["stranger"].locked is None  # still a choice by name


def test_the_ranking_and_the_fit_do_not_change_with_the_origin():
    packages = [_hf_package(repo="stranger/Nova-8B-GGUF"), _hf_package(repo="packager/Nova-8B-GGUF", quantization="Q8_0", weights_bytes=8 * GIB)]
    free = _build(packages)
    rated = _build(packages, _catalog_listing(["packager"]))

    def shape(doc):
        return [(e.rank, e.package_identity, e.fit, e.measurement_group) for e in doc.machines[0].ranked]

    assert shape(rated) == shape(free)
