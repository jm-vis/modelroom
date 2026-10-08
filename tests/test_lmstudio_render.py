"""AP9-LS-E: a whole render and its views for a machine with LM Studio (F7 a: one table, two list entries).

The laptop of `profiles_v4/laptop_lmstudio.json` names `["ollama", "lmstudio"]`: its table shows every Hugging
Face GGUF package once, with Ollama's fit, in one table without sections and without a prefix; the pull list
gives each of them a second entry for LM Studio. The same render with LM Studio alone, or with vLLM beside it,
and the existing goldens that must not move a byte.
"""

from __future__ import annotations

import json
from pathlib import Path

from modelroom.pull_choice import pull_rows
from modelroom.profile import HardwareProfile
from modelroom.render import document_json
from modelroom.room import GIB, Disk, LiveRoom
from modelroom.screen import FIT_HINT, plain
from modelroom.views import RunView, document_markdown, document_terminal, terminal_blocks

from test_lmstudio_weichen import _disk_place, _marked_in_markdown
from test_golden_vllm import FIXTURES, GOLDEN, _base_model, _compare, _gguf_package, _profile, _render, _tensor_package

MACHINE = "server"


def _laptop(keys=("ollama", "lmstudio")) -> HardwareProfile:
    data = json.loads((FIXTURES / "profiles_v4" / "laptop_lmstudio.json").read_text(encoding="utf-8"))
    data["runtimes"] = list(keys)
    return HardwareProfile.model_validate(data)


def _q8():
    package = _gguf_package()
    file = package.files[0].model_copy(update={"name": "Qwen3-8B-Q8_0.gguf", "size_bytes": 8_709_518_336})
    return package.model_copy(update={"quantization": "Q8_0", "files": [file]})


def _packages():
    return [_gguf_package(), _q8()]


def _document(keys=("ollama", "lmstudio"), packages=None):
    return _render(_laptop(keys), [_base_model()], packages if packages is not None else _packages(), context=8192, requests=1)


def _lines(document, **view) -> str:
    return "\n".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for=MACHINE, **view)))


# --- F7 (a): one table, two list entries ----------------------------------------------------------------------


def test_a_real_render_gives_one_table_row_and_two_unique_list_values_per_gguf_package():
    document = _document()
    block = document.machines[0]
    assert len(block.ranked) == 2 and {entry.fit.runtime for entry in block.ranked} == {"ollama"}
    rows = pull_rows(document, MACHINE, packages=_packages())
    assert [(row.rank, row.runtime) for row in rows] == [(1, "ollama"), (1, "lmstudio"), (2, "ollama"), (2, "lmstudio")]
    values = [row.value for row in rows]
    assert len(set(values)) == 4
    assert values[0].startswith("hf.co/unsloth/Qwen3-8B-GGUF:") and values[1].startswith("lmstudio unsloth/Qwen3-8B-GGUF@")
    assert [row.name for row in rows if row.runtime == "lmstudio"] == [
        "unsloth/Qwen3-8B-GGUF@" + entry.quantization for entry in block.ranked
    ]


def test_ollama_and_lmstudio_share_one_table_without_sections():
    document = _document()
    terminal, markdown = _lines(document), document_markdown(document)
    for text in (terminal, markdown):
        assert "with Ollama" not in text and "with LM Studio" not in text
        assert "One section per runtime" not in text
        assert "LM Studio ·" not in text and "LMS" not in text
    assert FIT_HINT[0] in markdown  # Ollama's key, the one a table of its rows carries


def test_a_machine_with_lmstudio_alone_sees_gguf_packages():
    document = _document(keys=("lmstudio",))
    block = document.machines[0]
    assert len(block.ranked) == 2
    rows = pull_rows(document, MACHINE, packages=_packages())
    assert {row.runtime for row in rows} == {"lmstudio"}
    # an Ollama-library package and a tensor package are not shown to it
    assert [entry.fit.runtime for entry in block.ranked] == ["ollama", "ollama"]


def test_a_machine_with_lmstudio_alone_keeps_the_ollama_form():
    """S6 (a): no vLLM row and no Ollama on the machine -- the table keeps Ollama's form (its `speed` line, its
    key), and nothing but the `runtime` row differs from the same machine with Ollama alone."""
    alone = document_terminal(_document(keys=("lmstudio",)))
    ollama = document_terminal(_document(keys=("ollama",)))
    without_runtime = "\n".join(line for line in alone.splitlines() if not line.startswith(" runtime"))
    assert "vLLM" not in alone
    assert without_runtime == ollama
    assert " runtime   LM Studio (from its profile)" in alone


def test_the_runtime_row_names_lm_studio():
    both = document_terminal(_document())
    assert " runtime   Ollama · LM Studio (from its profile)" in both
    assert " runtime" not in document_terminal(_document(keys=("ollama",)))  # Ollama alone says nothing new


def test_the_runtime_row_of_lmstudio_carries_no_vllm_note_for_unequal_cards():
    from modelroom.views_runtime import unequal_cards_note

    mixed = _laptop().model_copy(update={"runtimes": ["ollama", "lmstudio"]})
    assert unequal_cards_note(mixed) is None
    four = _profile().model_copy(update={"runtimes": ["ollama", "lmstudio"]})
    assert unequal_cards_note(four) is None
    with_vllm = _profile().model_copy(update={"runtimes": ["ollama", "vllm", "lmstudio"]})
    assert unequal_cards_note(with_vllm) == unequal_cards_note(_profile())


def test_with_vllm_the_runtime_row_names_all_three():
    profile = _profile().model_copy(update={"runtimes": ["ollama", "vllm", "lmstudio"]})
    document = _render(profile, [_base_model()], [_gguf_package(), _tensor_package()])
    assert " runtime   Ollama · vLLM · LM Studio (from its profile)" in document_terminal(document)


def test_an_empty_vllm_block_keeps_its_form():
    from test_views_sections import _vllm_only_profile

    document = _render(_vllm_only_profile(), [_base_model()], [_gguf_package()])
    assert document.machines[0].ranked == []
    _compare("golden_vllm_only_empty.txt", document_terminal(document) + "\n")


def test_the_head_of_a_machine_with_lmstudio_and_vllm_names_its_budgets_by_runtime():
    profile = _profile().model_copy(update={"runtimes": ["lmstudio", "vllm"]})
    document = _render(profile, [_base_model()], [_gguf_package(), _tensor_package()])
    head = " ".join(document_terminal(document).splitlines()[:8])
    assert "for LM Studio" in head and "for vLLM" in head and "for models" not in head and "for Ollama" not in head


# --- S3 (a): the disk piece of the room line -----------------------------------------------------------------


def _pieces(keys, ollama_drive="C:", lms_drive="C:", vllm_drive=None, lms_known=True):
    from modelroom.room_head import head_pieces

    document = _document(keys=keys)
    lmstudio = Disk(30 * GIB, lms_drive) if lms_known else Disk(None, None)
    vllm = None if vllm_drive is None else Disk(20 * GIB, vllm_drive)
    disk = Disk(int(41.2 * GIB), ollama_drive, lmstudio=lmstudio, vllm=vllm, runtimes=frozenset(keys))
    return [piece for piece in head_pieces(document.machines[0], LiveRoom(disk=disk), "this", frozenset(keys)) if piece.startswith("disk")]


def test_same_drive_gives_one_disk_piece_with_two_or_three_names():
    assert _pieces(["ollama", "lmstudio"]) == ["disk 41.2 GB free on C: (Ollama and LM Studio models)"]
    assert _pieces(["ollama", "lmstudio", "vllm"], vllm_drive="C:") == ["disk 41.2 GB free on C: (Ollama, LM Studio and vLLM models)"]


def test_other_drives_give_a_piece_for_each_place():
    assert _pieces(["ollama", "lmstudio"], lms_drive="D:") == [
        "disk 41.2 GB free on C: (Ollama models)",
        "disk 30.0 GB free on D: (LM Studio models)",
    ]
    assert _pieces(["ollama", "lmstudio", "vllm"], lms_drive="C:", vllm_drive="E:") == [
        "disk 41.2 GB free on C: (Ollama and LM Studio models)",
        "disk 20.0 GB free on E: (vLLM models)",
    ]


def test_the_line_for_ollama_and_vllm_stays_word_for_word():
    assert _pieces(["ollama", "vllm"], vllm_drive="C:") == ["disk 41.2 GB free on C: (Ollama and vLLM models)"]
    assert _pieces(["ollama", "vllm"], vllm_drive="E:") == [
        "disk 41.2 GB free on C: (Ollama models)",
        "disk 20.0 GB free on E: (vLLM models)",
    ]


def test_a_machine_with_lmstudio_alone_names_the_lmstudio_place_in_the_head():
    assert _pieces(["lmstudio"], lms_drive="D:") == ["disk 30.0 GB free on D: (LM Studio models)"]
    assert _pieces(["lmstudio"], lms_known=False) == ["disk unknown"]


def test_an_lmstudio_place_that_was_not_read_says_nothing_beside_ollama():
    assert _pieces(["ollama", "lmstudio"], lms_known=False) == ["disk 41.2 GB free on C: (Ollama models)"]


# --- S3 (a): the mark in a view and the package LM Studio holds ---------------------------------------------


def test_too_little_disk_at_the_lmstudio_place_marks_the_row():
    document = _document(keys=("lmstudio",))
    disk = Disk(100 * GIB, "C:", lmstudio=Disk(6 * GIB, "D:"), runtimes=frozenset({"lmstudio"}))
    terminal = _lines(document, live={MACHINE: LiveRoom(disk=disk)}, packages=_packages())
    markdown = document_markdown(document, {MACHINE: disk})
    assert _marked_in_markdown(markdown) == {2}  # the 8.1 GB Q8_0 file does not fit 6.0 GB, the 4.7 GB Q4_K_M does
    assert _disk_place(terminal) == _disk_place(markdown) == [("6.0 GB", "D:")]


def test_a_package_lm_studio_holds_is_marked_in_neither_terminal_nor_markdown():
    from modelroom.quantization import package_identity_key

    document = _document(keys=("ollama", "lmstudio"))
    held = frozenset(package_identity_key(package) for package in _packages())
    disk = Disk(2 * GIB, "C:", lmstudio=Disk(2 * GIB, "D:", held=held), runtimes=frozenset({"ollama", "lmstudio"}))
    view = {"lmstudio": held}
    terminal = _lines(document, live={MACHINE: LiveRoom(disk=disk)}, packages=_packages(), held=view)
    markdown = document_markdown(document, {MACHINE: disk})
    assert "more than the" not in terminal and "more than the" not in markdown
    assert " ×" not in markdown


# --- S2, S1 through a whole run: the registered servers --------------------------------------------------------


def test_the_start_row_and_the_note_of_step_4_come_through_the_table():
    from modelroom import local_servers
    from modelroom.runtimes import LMSTUDIO, Origins

    from fixture_lmstudio import lmstudio_servers, run_of

    run = run_of(lmstudio_servers(["lmstudio_v1_models_loaded.json"]))
    fact = LMSTUDIO.server.start_fact(Origins(daemon=None, servers=run.servers))
    assert fact is not None and fact.value == "LM Studio" and fact.note.startswith("1 loaded: ")
    assert [server.key for server in local_servers.runtimes.local_servers()] == ["ollama", "vllm", "lmstudio"]


def test_the_card_carries_no_lmstudio_row_in_the_negative_branch():
    from modelroom import local_servers

    from fixture_lmstudio import lmstudio_servers, run_of

    run = run_of(lmstudio_servers(["lmstudio_v1_models_loaded.json"]))
    assert [fact for fact in local_servers.runtimes.LMSTUDIO.server.card_facts(run)] == []


def test_default_transports_carries_the_lmstudio_transport_and_connects_nowhere():
    from modelroom.lmstudio_server import LocalLmStudioServer
    from modelroom.local_servers import default_transports

    transports = default_transports()
    assert isinstance(transports["lmstudio"], LocalLmStudioServer) and set(transports) == {"vllm", "lmstudio"}


def test_an_active_ssh_computer_reads_lmstudio_absent():
    from modelroom.remote_server import hello_result
    from modelroom.runtimes import present_runtimes

    from test_remote_server import _context

    assert "lmstudio" not in hello_result(_context())  # the `hello` of a server carries no field for it
    profile = _laptop(("ollama",))
    assert "lmstudio" not in present_runtimes(profile, "installed")  # vLLM may be there, LM Studio never is


# --- the goldens ---------------------------------------------------------------------------------------------


def test_the_markdown_of_a_laptop_with_lmstudio_reads_as_the_golden_file():
    _compare("golden_lmstudio_laptop.md", document_markdown(_document()))


def test_the_terminal_view_of_a_laptop_with_lmstudio_reads_as_the_golden_file():
    _compare("golden_lmstudio_laptop.txt", document_terminal(_document(), MACHINE) + "\n")


def test_the_json_of_a_laptop_with_lmstudio_reads_as_the_golden_file():
    _compare("golden_lmstudio_laptop.json", json.dumps(document_json(_document()), indent=2, sort_keys=True) + "\n")


def test_the_json_of_a_machine_with_lmstudio_writes_no_lmstudio_fit():
    blocks = document_json(_document())["machines"]
    assert all(entry["fit"].get("runtime", "ollama") == "ollama" for block in blocks for entry in block["ranked"])
    assert "lmstudio" in json.dumps(_laptop().model_dump(mode="json"))


# --- no existing output moves a byte -------------------------------------------------------------------------
# The goldens of vLLM, of the install blocks and of the guided screen are each compared with a freshly made
# output by the test module that owns them (`test_golden_vllm`, `test_views_sections`, `test_install_vllm`,
# `test_guided_screen`, `test_guided`); the proof that no file of them was touched by this branch is a diff
# against the base commit, made at the acceptance (`git diff --cached --stat 252c0cf -- tests/golden`), not a test.


def test_every_golden_file_is_referenced_by_a_test_module():
    """Portable, no git: each file of `tests/golden` is named in the text of some test module. This is a reference
    check only, not a comparison; the comparisons of the outputs stay in the modules that own the golden files."""
    sources = [path.read_text(encoding="utf-8") for path in Path(__file__).resolve().parent.glob("test_*.py")]
    unnamed = [path.name for path in GOLDEN.iterdir() if not any(path.name in source for source in sources)]
    assert unnamed == []
