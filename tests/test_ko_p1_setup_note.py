"""AP9-KO P1, part 4: the setup note of the model card over both vLLM commands (E4-B, V7; no switch of P3).

Where the catalog holds a note with status `instructions` for a model, every place that draws that model's two vLLM
commands -- the install line under the table, the choice in step 5, the handover for another computer -- puts the
red lines of `catalog_findings.setup_note_lines` before them, and the Markdown view carries the same lines. The
commands are not changed: no parser, backend or sampler switch is added from a card (that is P3, and never a
card-dependent switch). Without a note the lines are the lines of before.
"""

from __future__ import annotations

from pathlib import Path

from modelroom.catalog_findings import SETUP_NOTE, SETUP_NOTE_PACKAGE, SetupNote, setup_note_lines
from modelroom.dialog import FileAsker
from modelroom.guided_install import PullContext, pull_step
from modelroom.install_vllm import install_block_lines, vllm_install
from modelroom.pull_choice import ambiguous_names
from modelroom.screen import plain
from modelroom.views import RunView, terminal_blocks

from test_golden_vllm import BASE_REPO
from test_install_vllm import DOCKER, REPO, SERVE, _both, _fit, _gguf_package, _ollama_only, _package, _profile, _run

REVISION = "d" * 40


def _note(*, package: str | None = None, status: str = "instructions") -> SetupNote:
    read = status != "not_read"
    return SetupNote.model_validate(
        {
            "model": BASE_REPO, "status": status, "card_revision": REVISION if read else None, "checked": "2026-10-13",
            "source_url": f"https://huggingface.co/{BASE_REPO}/blob/{REVISION}/README.md" if read else f"https://huggingface.co/{BASE_REPO}",
            "package": package if status == "instructions" else None,
            "reason": None if read else "the card did not answer",
        }
    )


def _with_note(document, note: SetupNote | None):
    return document.model_copy(update={"setup_notes": {} if note is None else {BASE_REPO: note}})


def _vllm_text(document, extra=None) -> list[str]:
    view = RunView(install_for="server", packages=[_gguf_package(), _package()], **(extra or {}))
    return [plain(line) for line in terminal_blocks(document, view=view)]


def _first_lines(note: SetupNote) -> list[str]:
    return list(setup_note_lines(note))


# --- the one place: install_block_lines ----------------------------------------------------------------------------


def test_the_lines_of_the_note_stand_over_both_commands_in_the_warning_style():
    profile, package = _profile(), _package()
    install = vllm_install(package, _fit(profile, package), profile)
    note = _note(package="nova-inference")

    with_note = install_block_lines("install", 2, install, repo=REPO, setup_note=note)
    without = install_block_lines("install", 2, install, repo=REPO)

    text = [plain(line) for line in with_note]
    sentences = _first_lines(note)
    assert sentences == [SETUP_NOTE.format(model=BASE_REPO, url=note.source_url), SETUP_NOTE_PACKAGE.format(package="nova-inference")]
    assert [row.strip().removeprefix("install").strip() for row in text[:2]] == sentences
    styles = [style for line in with_note[:2] for style, part in line if part in sentences]
    assert styles == ["class:warn", "class:warn"]
    # behind them, both commands and everything after: the lines of before, but for the label that moved up
    assert [row.strip().removeprefix("install").strip() for row in text[2:]] == [row.strip().removeprefix("install").strip() for row in (plain(line) for line in without)]
    assert SERVE in text[2] and DOCKER in text[3]


def test_a_note_without_instructions_or_without_a_note_changes_nothing():
    profile, package = _profile(), _package()
    install = vllm_install(package, _fit(profile, package), profile)
    plain_lines = install_block_lines("install", 2, install, repo=REPO)

    assert install_block_lines("install", 2, install, repo=REPO, setup_note=None) == plain_lines
    for status in ("none_recognized", "not_read"):
        assert install_block_lines("install", 2, install, repo=REPO, setup_note=_note(status=status)) == plain_lines


def test_a_row_without_a_command_gets_no_note_lines():
    from modelroom.install_vllm import VllmInstall

    lines = install_block_lines("install", 2, VllmInstall(None, reason="no safe command"), repo=REPO, setup_note=_note())
    assert [plain(line) for line in lines] == [" install   #2 no safe command"]


# --- the views ---------------------------------------------------------------------------------------------------------


def test_the_table_carries_the_note_over_both_commands():
    note = _note(package="nova-inference")
    text = _vllm_text(_with_note(_both(), note))

    first = next(index for index, line in enumerate(text) if _first_lines(note)[0] in line)
    assert _first_lines(note)[1] in text[first + 1]
    assert SERVE in text[first + 2] and DOCKER in text[first + 3]
    # no switch is added to either command
    assert all(switch not in line for line in text for switch in ("--attention-backend", "--reasoning-parser", "--tool-call-parser", "VLLM_USE_FLASHINFER_SAMPLER"))


def test_without_a_note_the_table_is_the_table_of_before():
    base = _vllm_text(_with_note(_both(), None))
    index = next(index for index, line in enumerate(base) if line.startswith(" install   #2  "))

    assert base[index] == f" install   #2  {SERVE}" and base[index + 1] == f"     docker    {DOCKER}"
    assert not any("setup instructions" in line for line in base)


def test_the_handover_for_another_computer_carries_the_note(tmp_path: Path):
    from modelroom.pull_other import pull_other_step

    note = _note()
    document = _with_note(_both(), note)
    run, lines = _run(tmp_path, FileAsker({"pull_other": f"server vllm {REPO}"}))

    pull_other_step(run, PullContext(document, "laptop", [_gguf_package(), _package()], ambiguous_names([_gguf_package(), _package()])))

    text = "\n".join(lines)
    assert _first_lines(note)[0] in text and text.index(_first_lines(note)[0]) < text.index("vllm serve")
    assert "modelroom pull" not in text


def test_the_choice_in_step_5_carries_the_note(tmp_path: Path):
    note = _note(package="nova-inference")
    document = _with_note(_both(), note)
    packages = [_gguf_package(), _package()]
    run, lines = _run(tmp_path, FileAsker({"pull_package": f"vllm {REPO}"}))

    pull_step(run, PullContext(document, "server", packages, ambiguous_names(packages)), True)

    text = "\n".join(lines)
    assert all(sentence in text for sentence in _first_lines(note))
    assert text.index(_first_lines(note)[0]) < text.index("vllm serve") < text.index("docker")


def test_the_markdown_view_carries_the_note_for_a_model_with_a_vllm_row_and_the_json_view_the_note_itself():
    from modelroom.render import document_json
    from modelroom.views import document_markdown

    note = _note(package="nova-inference")
    document = _with_note(_both(), note)

    markdown = document_markdown(document)
    assert all(sentence in markdown for sentence in _first_lines(note))
    assert document_json(document)["setup_notes"][BASE_REPO]["package"] == "nova-inference"
    # a document without a note is the Markdown of before, and a model with no vLLM row shows none
    assert "setup instructions" not in document_markdown(_with_note(_both(), None))
    assert "setup instructions" not in document_markdown(_with_note(_ollama_only(), note))
