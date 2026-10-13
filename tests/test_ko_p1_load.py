"""AP9-KO P1, part 1: an `absent` check locks the installation of its package (L1, V2, V5, V11).

Only `absent` blocks. The row keeps its place in the ranking; the views say why under the table, the install line
passes it, the pull list locks it for every runtime that loads a GGUF file (Ollama, LM Studio) -- an explicit choice
included -- and a package the fit would set aside as `no_fit` is carried as `not_covered` with the reason. No model,
publisher or packager is named as a case: the rule reads the check on the row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelroom.catalog_findings import PackageCheck, load_blocked_text
from modelroom.daemon import PULL_PATH
from modelroom.dialog import AnswerInvalidError
from modelroom.guided_install import PullContext, pull_step
from modelroom.pull_choice import ambiguous_names, pull_rows
from modelroom.screen import plain
from modelroom.views import RunView, document_markdown, terminal_blocks, terminal_lines

from fixture_rows import document, package
from test_ko_render import DAY, _build, _catalog, _check_for
from test_pull_choice import _daemon, _Recording, _run, _with_machine_runtimes
from test_render import GIB, _hf_package

REPO = "acme/Nova-8B"


def _terminal(doc, packages) -> str:
    view = RunView(install_for="workstation", packages=packages)
    return "\n".join(plain(line) for line in terminal_blocks(doc, view=view))


def _blocked(*packages):
    """The render document (a real build) of `packages`, a catalog checking each of them `absent`."""
    return _build(list(packages), _catalog([_check_for(p) for p in packages]))


def _text_for(check_dict: dict, runtime: str = "Ollama") -> str:
    return load_blocked_text(runtime, REPO, PackageCheck.model_validate(check_dict))


# --- the fit: not_covered with the reason, not no_fit ---------------------------------------------------------------


def test_a_package_the_fit_sets_aside_as_no_fit_is_not_covered_with_the_reason_when_a_check_blocks_it():
    huge = _hf_package(weights_bytes=900 * GIB)
    plain_block = _build([huge]).machines[0]
    assert len(plain_block.no_fit) == 1 and plain_block.not_covered == []  # control: without evidence it is no_fit

    check = _check_for(huge)
    block = _build([huge], _catalog([check])).machines[0]

    assert block.no_fit == [] and len(block.not_covered) == 1
    entry = block.not_covered[0]
    assert entry.reason == _text_for(check) and entry.fit.fit_class == "unknown"
    assert entry.note.code == "load_blocked" and entry.note.text == entry.reason
    assert entry.load_check is not None and entry.load_check.result == "absent"


@pytest.mark.parametrize("variant", [{"result": "present"}, {"revision": "b" * 40}])
def test_a_check_that_does_not_block_leaves_the_no_fit_as_it_is(variant):
    """`present` is no block, and a check of another revision is no finding for this package: `unknown`, not `absent`."""
    huge = _hf_package(weights_bytes=900 * GIB)
    block = _build([huge], _catalog([_check_for(huge, **variant)])).machines[0]

    assert len(block.no_fit) == 1 and block.not_covered == []


def test_a_blocked_row_that_fits_keeps_its_place_class_and_rank():
    q4, q8 = _hf_package(), _hf_package(quantization="Q8_0", weights_bytes=8 * GIB)
    free = _build([q4, q8]).machines[0]
    blocked = _build([q4, q8], _catalog([_check_for(q4)])).machines[0]

    def shape(block):
        return [(e.rank, e.package_identity, e.fit.fit_class, e.fit.need_gib) for e in block.ranked]

    assert shape(blocked) == shape(free)  # the ranking is the ranking of 1.0.0; the block is the views' and the list's


# --- the table says why ---------------------------------------------------------------------------------------------


def test_the_table_names_a_blocked_row_in_the_guided_result_and_in_the_stand_alone_render_alike():
    q4, q8 = _hf_package(), _hf_package(quantization="Q8_0", weights_bytes=8 * GIB)
    doc = _build([q4, q8], _catalog([_check_for(q4)]))
    rank = next(e.rank for e in doc.machines[0].ranked if e.quantization == "Q4_K_M")
    expected = f"#{rank} {_text_for(_check_for(q4))}"

    guided = _terminal(doc, [q4, q8])
    render = "\n".join(terminal_lines(doc))  # `modelroom render`: no machine of the run, no install line
    markdown = document_markdown(doc)

    assert any(expected in line and line.strip().startswith("blocked") for line in guided.splitlines())
    assert any(expected in line and line.strip().startswith("blocked") for line in render.splitlines())
    assert f"blocked: {expected}" in markdown
    assert "ollama pull" not in render  # the stand-alone render draws no install line, with or without evidence


def test_a_document_without_a_block_draws_no_blocked_line():
    q4 = _hf_package()
    for doc in (_build([q4]), _build([q4], _catalog([_check_for(q4, result="present")]))):
        assert "blocked" not in "\n".join(terminal_lines(doc)) and "blocked:" not in document_markdown(doc)


def _red_lines(text: str, expected: str) -> list[str]:
    return [line for line in text.splitlines() if expected in line and line.strip().startswith("blocked")]


def test_a_blocked_package_the_fit_set_aside_as_no_fit_keeps_the_red_line_in_both_views():
    """Fix round 1 (b1 MUSS): the package is `not_covered` with the one reason, which the `shown` line then leaves
    out; the table must still say the L1 words, in the terminal and in the Markdown file."""
    huge = _hf_package(weights_bytes=900 * GIB)
    check = _check_for(huge)
    doc = _build([huge], _catalog([check]))
    expected = _text_for(check)

    assert doc.machines[0].ranked == [] and len(doc.machines[0].not_covered) == 1
    assert _red_lines("\n".join(terminal_lines(doc)), expected)
    assert any(line.startswith("blocked:") and expected in line for line in document_markdown(doc).splitlines())


def test_two_blocked_packages_set_aside_under_the_same_packager_and_quantization_are_told_apart():
    """Fix round 2: the label `packager · quant` is extended by the repository only where two would read alike."""
    a = _hf_package(weights_bytes=900 * GIB, repo="packager/Nova-8B-GGUF")
    b = _hf_package(weights_bytes=900 * GIB, repo="packager/Nova-8B-Extra-GGUF")
    doc = _build([a, b], _catalog([_check_for(a), _check_for(b)]))
    text = "\n".join(terminal_lines(doc))
    labels = [line.split("blocked", 1)[-1].strip().split(" Ollama", 1)[0] for line in text.splitlines() if "installation not offered" in line]

    assert len(labels) == 2 and len(set(labels)) == 2
    assert any("packager/Nova-8B-GGUF" in label for label in labels) and any("packager/Nova-8B-Extra-GGUF" in label for label in labels)
    # control: one blocked package keeps the short label
    single = _build([a], _catalog([_check_for(a)]))
    assert any(line.strip().startswith("blocked") and "packager · Q4_K_M Ollama" in line for line in terminal_lines(single))


def test_two_file_identities_of_one_repository_and_quantization_are_told_apart_by_the_file_and_keep_their_order():
    """Last stage of the label: same packager, same quantization, same repository, two file identities."""
    from modelroom.pull_choice import blocked_lines_of

    a = _hf_package(weights_bytes=900 * GIB)
    b = a.model_copy(update={"files": [a.files[0].model_copy(update={"name": "Q4_K_M/Nova-8B-Q4_K_M.gguf"})]})
    doc = _build([a, b], _catalog([_check_for(a), _check_for(b)]))
    block = doc.machines[0]

    lines = blocked_lines_of(block)
    identities = [entry.package_identity[2] for entry in block.not_covered]

    assert len(lines) == 2 and len({label for label, _text in lines}) == 2
    assert all(identity in label for (label, _text), identity in zip(lines, identities))  # the order of the table
    assert all("packager/Nova-8B-GGUF" in label for label, _text in lines)


def test_a_blocked_package_the_fit_could_not_judge_anyway_keeps_the_red_line_in_both_views():
    """An `unknown` fit for another reason (here a weight file without a size) keeps its own reason and shows the block."""
    unsized = _hf_package(weights_bytes=0)
    check = _check_for(unsized)
    doc = _build([unsized], _catalog([check]))
    entry = doc.machines[0].not_covered[0]
    expected = _text_for(check)

    assert entry.reason != expected and entry.load_check is not None
    assert _red_lines("\n".join(terminal_lines(doc)), expected)
    assert any(line.startswith("blocked:") and expected in line for line in document_markdown(doc).splitlines())
    # control: the same package without a block says nothing of the kind
    free = _build([unsized], _catalog([_check_for(unsized, result="present")]))
    assert "blocked" not in "\n".join(terminal_lines(free)) and "blocked:" not in document_markdown(free)


# --- the install line passes the row --------------------------------------------------------------------------------


def test_the_install_line_offers_the_next_row_not_the_blocked_one():
    q4, q8 = _hf_package(), _hf_package(quantization="Q8_0", weights_bytes=8 * GIB)
    doc = _build([q4, q8], _catalog([_check_for(q4)]))

    text = _terminal(doc, [q4, q8])

    assert "ollama pull hf.co/packager/Nova-8B-GGUF:Q8_0" in text
    assert "Q4_K_M" not in "\n".join(line for line in text.splitlines() if "ollama pull" in line)


def test_where_every_shown_row_is_blocked_nothing_is_offered_and_the_line_says_so():
    from modelroom.catalog_findings import NO_RECOMMENDATION

    q4, q8 = _hf_package(), _hf_package(quantization="Q8_0", weights_bytes=8 * GIB)
    doc = _blocked(q4, q8)

    text = _terminal(doc, [q4, q8])

    assert "ollama pull" not in text
    assert any(line.strip().startswith("install") and NO_RECOMMENDATION in line for line in text.splitlines())


# --- the pull list locks it, an explicit choice included ------------------------------------------------------------


def test_the_pull_row_of_a_blocked_package_is_locked_with_the_red_line():
    q4, q8 = _hf_package(), _hf_package(quantization="Q8_0", weights_bytes=8 * GIB)
    doc = _build([q4, q8], _catalog([_check_for(q4)]))

    rows = {row.name.rsplit(":", 1)[1]: row for row in pull_rows(doc, "workstation", packages=[q4, q8])}

    assert rows["Q4_K_M"].locked == _text_for(_check_for(q4)) == rows["Q4_K_M"].blocked
    assert rows["Q8_0"].locked is None and rows["Q8_0"].blocked is None


def test_an_explicit_choice_of_a_blocked_row_is_refused_and_downloads_nothing(tmp_path: Path):
    q4 = _hf_package()
    doc = _blocked(q4)
    name = "hf.co/packager/Nova-8B-GGUF:Q4_K_M"
    for answer in (name, "#1 locked"):
        asker = _Recording({"pull_package": answer, "pull": True})
        daemon = _daemon()
        run, _lines = _run(tmp_path / answer.replace(" ", "_").replace("/", "_").replace(":", "_"), asker, daemon)

        with pytest.raises(AnswerInvalidError):
            pull_step(run, PullContext(doc, "workstation", [q4], ambiguous_names([q4])), True)
        assert not any(call[1] == PULL_PATH for call in daemon.calls)


def test_lm_studio_is_locked_too_and_a_row_of_another_account_is_not_affected():
    q4 = _hf_package()
    other = _hf_package(quantization="Q8_0", weights_bytes=8 * GIB)
    doc = _with_machine_runtimes(_build([q4, other], _catalog([_check_for(q4)])), "workstation", ["ollama", "lmstudio"])

    rows = pull_rows(doc, "workstation", packages=[q4, other])
    by = {(row.runtime, row.cell.rsplit(" · ", 1)[1]): row for row in rows}

    check = _check_for(q4)
    assert by[("ollama", "Q4_K_M")].locked == _text_for(check, "Ollama")
    assert by[("lmstudio", "Q4_K_M")].locked == _text_for(check, "LM Studio")
    assert by[("ollama", "Q8_0")].locked is None and by[("lmstudio", "Q8_0")].locked is None
    # the table says it once, for both runtimes
    assert _text_for(check, "Ollama and LM Studio") in "\n".join(terminal_lines(doc))


def test_a_blocked_package_of_the_fixture_documents_is_locked_without_the_render():
    """The lock reads the row's check, so a document handed over (a JSON view read back) locks the same way."""
    pkg = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
    doc = document([pkg])
    entry = doc.machines[0].ranked[0]
    check = PackageCheck.model_validate(
        {
            "repo": pkg.repo, "revision": pkg.revision or "a" * 40, "identity": "Nova-7B-Q4_K_M.gguf", "file": "Nova-7B-Q4_K_M.gguf",
            "quantization": "Q4_K_M", "architecture": "nova7", "architecture_source": "file_header", "result": "absent",
            "llamacpp_commit": "45451c366d4256f94fbc8845645e31a6b9155054",
            "source_url": "https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/src/llama-arch.cpp",
            "checked": DAY.isoformat(),
        }
    )
    marked = doc.model_copy(
        update={"machines": [doc.machines[0].model_copy(update={"ranked": [entry.model_copy(update={"load_check": check})]})]}
    )

    [row] = pull_rows(marked, "workstation")

    assert row.locked == load_blocked_text("Ollama", entry.base_model_hf_repo, check)
