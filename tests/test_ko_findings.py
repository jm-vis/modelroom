"""Contract tests of AP9-KO P0, part 1: the finding forms, the source of the architecture block, the
scope rule of the Hub's `gguf` field, the origin rule and the fixed wordings (CONTRACTS.md, "Package
findings and the load check").

The block of llama.cpp and the Hub answers below are the recorded originals of the two feasibility
proofs (`docs/ap9-ko/feasibility/`), not rebuilt by hand. The synthetic parts (names of models,
packagers) follow the shape of the real cases and name nobody.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom import catalog_findings as F
from modelroom.http import _ALLOWED_HTTPS_HOSTS

REPO = Path(__file__).resolve().parent.parent
FEASIBILITY = REPO / "docs" / "ap9-ko" / "feasibility"
COMMIT = "45451c366d4256f94fbc8845645e31a6b9155054"  # the llama.cpp commit the recorded block comes from
REVISION = "a" * 40
OTHER_REVISION = "b" * 40
DAY = date(2026, 10, 13)


FULL_FILE = f"llama-arch.cpp.{COMMIT}.txt"  # the whole file src/llama-arch.cpp at COMMIT, as recorded


def _block_text() -> str:
    """The whole recorded file: what the scanner reads, and the base of every changed case below."""
    return (FEASIBILITY / FULL_FILE).read_text(encoding="utf-8")


def _cut_text() -> str:
    """The first recording of proof 2, cut to the block and the lookup."""
    return (FEASIBILITY / "v5-llamacpp-arch-block.txt").read_text(encoding="utf-8")


def _hub_records() -> list[dict]:
    return json.loads((FEASIBILITY / "v1-package-scope-hf-responses.json").read_text(encoding="utf-8"))


def _check(**changes) -> dict:
    data = {
        "repo": "packager/Nova-9B-GGUF",
        "revision": REVISION,
        "identity": "Nova-9B-Q8_0.gguf",
        "file": "Nova-9B-Q8_0.gguf",
        "quantization": "Q8_0",
        "architecture": "nova9",
        "architecture_source": "file_header",
        "result": "absent",
        "llamacpp_commit": COMMIT,
        "source_url": F.llamacpp_page_url(COMMIT),
        "checked": DAY,
    }
    data.update(changes)
    return data


def _note(**changes) -> dict:
    data = {
        "model": "acme/Nova-9B",
        "status": "instructions",
        "card_revision": REVISION,
        "source_url": f"https://huggingface.co/acme/Nova-9B/blob/{REVISION}/README.md",
        "checked": DAY,
        "package": None,
        "reason": None,
    }
    data.update(changes)
    return data


# --- V5: the canonical source, the commit, the address ------------------------------------------------


def test_the_canonical_source_is_one_repository_one_file_and_one_host():
    assert (F.LLAMACPP_REPO, F.LLAMACPP_ARCH_FILE, F.LLAMACPP_HOST) == ("ggml-org/llama.cpp", "src/llama-arch.cpp", "api.github.com")
    assert F.llamacpp_commit_url() == "https://api.github.com/repos/ggml-org/llama.cpp/commits/HEAD"
    assert F.llamacpp_block_url(COMMIT) == f"https://api.github.com/repos/ggml-org/llama.cpp/contents/src/llama-arch.cpp?ref={COMMIT}"
    assert F.llamacpp_page_url(COMMIT) == f"https://github.com/ggml-org/llama.cpp/blob/{COMMIT}/src/llama-arch.cpp"


def test_the_two_answers_have_fixed_limits_and_no_redirect():
    assert F.LLAMACPP_COMMIT_ACCEPT == "application/vnd.github.sha"
    assert F.LLAMACPP_FILE_ACCEPT == "application/vnd.github.raw+json"
    assert (F.LLAMACPP_COMMIT_MAX_BYTES, F.LLAMACPP_FILE_MAX_BYTES, F.LLAMACPP_MAX_REDIRECTS) == (256, 512 * 1024, 0)


def test_the_runtime_of_modelroom_gets_no_new_host():
    assert F.LLAMACPP_HOST not in _ALLOWED_HTTPS_HOSTS
    assert _ALLOWED_HTTPS_HOSTS == frozenset({"huggingface.co", "ollama.com", "registry.ollama.ai"})


@pytest.mark.parametrize("bad", ["HEAD", "master", "", COMMIT[:39], COMMIT.upper(), "g" * 40, COMMIT + "0"])
def test_only_a_resolved_commit_names_a_file(bad):
    with pytest.raises(ValueError):
        F.llamacpp_block_url(bad)
    with pytest.raises(ValueError):
        F.llamacpp_page_url(bad)


@pytest.mark.parametrize("text", [COMMIT, "HEAD", COMMIT + "\n", "x" * 40, COMMIT.upper()])
def test_the_commit_answer_is_forty_lowercase_hex_characters_and_nothing_else(text):
    if text == COMMIT:
        assert F.read_commit(text) == COMMIT
    else:
        with pytest.raises(F.ArchBlockError):
            F.read_commit(text)


# --- V5: the block, read whole or not at all --------------------------------------------------------------


def test_the_recorded_block_is_understood_whole():
    names = F.parse_arch_names(_block_text())
    assert len(names) == 157
    assert F.parse_arch_names(_cut_text()) == names  # the cut recording says the same
    assert {"llama", "clip", "glm4moe", "bailingmoe2", "k2-horizon"} <= names
    assert "(unknown)" not in names  # the sentinel is no architecture


def test_what_the_recorded_commit_does_not_know_is_absent_and_what_it_knows_is_present():
    names = F.parse_arch_names(_block_text())
    assert F.compare_architecture("llama", names) == "present"
    assert F.compare_architecture("an-architecture-nobody-wrote", names) == "absent"


def test_the_comparison_is_the_whole_identifier_and_nothing_else():
    names = frozenset({"llama", "llama4"})
    assert F.compare_architecture("llama", names) == "present"
    assert F.compare_architecture("llam", names) == "absent"
    assert F.compare_architecture("LLAMA", names) == "absent"
    assert F.compare_architecture("llama ", names) == "absent"


LOOKUP_HEAD = "llm_arch llm_arch_from_string(const std::string & name) {"


def _lines() -> list[str]:
    return _block_text().split("\n")


def _without(predicate) -> str:
    return "\n".join(line for line in _lines() if not predicate(line))


def _replaced(old: str, new: str, count: int = 1) -> str:
    text = _block_text()
    assert old in text
    return text.replace(old, new, count)


BLOCK_FAULTS = {
    "empty": lambda: "",
    "no start": lambda: _replaced("LLM_ARCH_NAMES = {", "LLM_ARCH_LABELS = {"),
    "never closed": lambda: _replaced('"(unknown)"        },\n};\n', '"(unknown)"        },\n'),
    "unknown entry form": lambda: _replaced('{ LLM_ARCH_LLAMA,            "llama"            },', '{ LLM_ARCH_LLAMA, "llama", 3 },'),
    "macro instead of an entry": lambda: _replaced('{ LLM_ARCH_LLAMA,            "llama"            },', 'LLM_ARCH_ENTRY(LLAMA, "llama")'),
    "name in single quotes": lambda: _replaced('"llama4"', "'llama4'"),
    "duplicate name": lambda: _replaced('"llama4"', '"llama"'),
    "duplicate enum": lambda: _replaced("LLM_ARCH_LLAMA4,", "LLM_ARCH_LLAMA,"),
    "the lookup no longer compares the name": lambda: _replaced("if (kv.second == name) {", "if (kv.first == name) {"),
    "the lookup is gone": lambda: _replaced("llm_arch llm_arch_from_string(const std::string & name) {", "llm_arch other_lookup(const std::string & name) {"),
    "the block twice": lambda: _block_text() + "\n" + "\n".join(_lines()[7:167]),
    "extra logic before the loop of the lookup": lambda: _replaced(
        "    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT",
        '    if (name == "future_arch") { return LLM_ARCH_LLAMA; }\n    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT',
    ),
    "extra logic after the loop of the lookup": lambda: _replaced(
        "    return LLM_ARCH_UNKNOWN;", '    if (name == "future_arch") { return LLM_ARCH_LLAMA; }\n    return LLM_ARCH_UNKNOWN;'
    ),
    "the lookup returns something else": lambda: _replaced("            return kv.first;", "            return LLM_ARCH_LLAMA;"),
    "the lookup is cut off": lambda: _replaced("\n    return LLM_ARCH_UNKNOWN;\n}", "\n"),
    "far fewer entries": lambda: _without(lambda line: line.startswith("    { LLM_ARCH_") and "LLAMA" not in line and "UNKNOWN" not in line),
    # Fix round 2 (sweep of the class "a part understood is taken for the whole"): each case below is a change
    # the compiler reads differently from a line-by-line reading, so a name would be wrongly present or absent.
    "code on the head line of the loop of the lookup": lambda: _replaced(
        "    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT",
        '    for (const auto & kv : LLM_ARCH_NAMES) { if (name == "future_arch") return LLM_ARCH_LLAMA; // NOLINT',
    ),
    "another comment on the head line of the loop": lambda: _replaced(
        "    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT", "    for (const auto & kv : LLM_ARCH_NAMES) { /* NOLINT */"
    ),
    "a lone carriage return starts a line the reading does not see": lambda: _replaced(
        '"llama"            },\n    { LLM_ARCH_LLAMA4', '"llama"            }, // note\r    { LLM_ARCH_LLAMA4'
    ),
    "a backslash ends a trailing comment and splices the next entry into it": lambda: _replaced(
        '"llama"            },\n', '"llama"            }, // note \\\n'
    ),
    "a backslash ends a comment line and splices the next entry into it": lambda: _replaced(
        "\n    { LLM_ARCH_LLAMA4", "\n    // note\\ \n    { LLM_ARCH_LLAMA4"
    ),
    "the block inside a block comment": lambda: _replaced('"(unknown)"        },\n};\n', '"(unknown)"        },\n};\n*/\n').replace(
        "static const std::map", "/*\nstatic const std::map", 1
    ),
    "a second lookup under a preprocessor condition": lambda: _replaced(
        "llm_arch llm_arch_from_string(const std::string & name) {",
        "#if 1\nllm_arch llm_arch_from_string(const std::string & name) {\n"
        '    if (name == "future_arch") { return LLM_ARCH_LLAMA; }\n    return LLM_ARCH_UNKNOWN;\n}\n#else\n'
        "llm_arch llm_arch_from_string(const std::string & name) {",
    )
    + "\n#endif\n",
    "a macro renames the block": lambda: _replaced(LOOKUP_HEAD, "#define LLM_ARCH_NAMES LLM_ARCH_NAMES_V2\n" + LOOKUP_HEAD),
    # Fix round 3 (decision of 2026-10-13: realistic changes of form, no C++ parser): exotic forms are refused whole.
    "a digraph of the hash sign": lambda: _replaced(LOOKUP_HEAD, "%:define LLM_ARCH_NAMES LLM_ARCH_NAMES_V2\n" + LOOKUP_HEAD),
    "a form feed in a preprocessor line": lambda: _replaced(LOOKUP_HEAD, "#\fdefine LLM_ARCH_NAMES LLM_ARCH_NAMES_V2\n" + LOOKUP_HEAD),
    "a vertical tab in a preprocessor line": lambda: _replaced(LOOKUP_HEAD, "#\vdefine LLM_ARCH_NAMES LLM_ARCH_NAMES_V2\n" + LOOKUP_HEAD),
    "the old block inside a raw string beside a new map on one line": lambda: _replaced(
        "static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {",
        'static const char * OLD_ARCH_NAMES = R"arch(\nstatic const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {',
    ).replace(
        '"(unknown)"        },\n};\n',
        '"(unknown)"        },\n};\n)arch";\n'
        'static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = { { LLM_ARCH_LLAMA, "llama" }, { LLM_ARCH_UNKNOWN, "(unknown)" } };\n',
        1,
    ),
    "a raw string with a prefix": lambda: _replaced(LOOKUP_HEAD, 'static const auto * NOTE = u8R"(x)";\n' + LOOKUP_HEAD),
    "an early end of the block leaves entries behind": lambda: _replaced(
        '"k2-horizon"       },\n', '"k2-horizon"       },\n};\n    { LLM_ARCH_LATE,             "late"             },\n'
    ),
    "an entry after the end of the block": lambda: _replaced(
        '    { LLM_ARCH_K2_HORIZON,       "k2-horizon"       },\n    { LLM_ARCH_UNKNOWN,          "(unknown)"        },\n};\n',
        '    { LLM_ARCH_UNKNOWN,          "(unknown)"        },\n};\n    { LLM_ARCH_K2_HORIZON,       "k2-horizon"       },\n};\n',
    ),
    "the sentinel is not the last entry": lambda: _replaced(
        '    { LLM_ARCH_UNKNOWN,          "(unknown)"        },\n', ""
    ).replace('    { LLM_ARCH_LLAMA,            "llama"            },', '    { LLM_ARCH_UNKNOWN,          "(unknown)"        },\n    { LLM_ARCH_LLAMA,            "llama"            },', 1),
    # Already rejected before this round; kept as regressions of the same class.
    "a preprocessor line inside the block": lambda: _replaced("\n    { LLM_ARCH_LLAMA4", "\n#if 0\n    { LLM_ARCH_LLAMA4"),
    "a block comment inside the block": lambda: _replaced("\n    { LLM_ARCH_LLAMA4", "\n    /* { LLM_ARCH_LLAMA4"),
    "code after the comma of an entry": lambda: _replaced('"llama"            },', '"llama"            }, { LLM_ARCH_LLAMA4, "llama4" },'),
}


@pytest.mark.parametrize("name", sorted(BLOCK_FAULTS))
def test_a_block_that_is_not_understood_whole_is_a_fault_never_a_smaller_result(name):
    with pytest.raises(F.ArchBlockError):
        F.parse_arch_names(BLOCK_FAULTS[name]())


def test_a_comment_or_a_blank_line_inside_the_block_changes_nothing():
    marked = _replaced('    { LLM_ARCH_LLAMA,            "llama"            },', '    // the first one\n\n    { LLM_ARCH_LLAMA,            "llama"            },')
    assert F.parse_arch_names(marked) == F.parse_arch_names(_block_text())


@pytest.mark.parametrize("comment", ["// NOLINT(performance-for-range-copy)", "// every entry, in the order of the enum", ""])
def test_an_ordinary_comment_or_none_on_the_head_line_of_the_loop_changes_nothing(comment):
    """Codex b4: a more specific `NOLINT` or another `//` comment after the exactly checked head ` {` is a
    realistic change; the exotic forms and a backslash at the end of the line stay faults."""
    head = "    for (const auto & kv : LLM_ARCH_NAMES) {"
    changed = _replaced(head + " // NOLINT", (head + " " + comment).rstrip())
    assert F.parse_arch_names(changed) == F.parse_arch_names(_block_text())
    assert F.parse_arch_names(changed.replace("\n", "\r\n")) == F.parse_arch_names(_block_text())


def test_windows_line_ends_change_nothing():
    assert F.parse_arch_names(_block_text().replace("\n", "\r\n")) == F.parse_arch_names(_block_text())


def test_the_whole_recorded_file_and_realistic_changes_of_it_are_read():
    """The whole file at the recorded commit, and changes the developers of llama.cpp may make to it."""
    whole = F.parse_arch_names(_block_text())
    assert len(whole) == 157
    assert F.parse_arch_names(_replaced('"llama"            },', '"llama"            },  \t')) == whole
    added = F.parse_arch_names(
        _replaced("    { LLM_ARCH_UNKNOWN,", '    { LLM_ARCH_NOVA9,            "nova9"            },\n    { LLM_ARCH_UNKNOWN,')
    )
    assert len(added) == 158 and added - whole == {"nova9"}
    assert len(F.parse_arch_names(_block_text().replace("\n", "\r\n"))) == 157


def test_the_plausible_floor_of_entries_is_a_hundred():
    assert F.ARCH_MIN_ENTRIES == 100


# --- V1: what the Hub's `gguf` field says, and of which file ------------------------------------------------


def test_the_recorded_answers_cover_seven_repositories_of_three_base_models():
    records = _hub_records()
    assert len(records) == 7
    assert len({record["base_model"] for record in records}) == 3


def test_the_two_recorded_proofs_agree_on_what_the_hub_states_and_the_source_knows():
    """The identifiers the Hub states for the recorded repositories, compared with the recorded block."""
    names = F.parse_arch_names(_block_text())
    states = {record["model_gguf_field"]["architecture"] for record in _hub_records()}
    assert {architecture: F.compare_architecture(architecture, names) for architecture in sorted(states)} == {
        "bailingmoe2": "present", "glm4moe": "present", "kolibri1": "absent", "sarvam_moe": "absent",
    }


@pytest.mark.parametrize("record", _hub_records() if FEASIBILITY.exists() else [], ids=lambda record: record["repo"])
def test_the_hub_field_describes_the_one_file_whose_size_is_total_file_size(record):
    files = [(f["path"], f["size"]) for f in record["gguf_files"]]
    found = F.hub_metadata_file(files, record["model_gguf_field"]["totalFileSize"])
    assert found is not None and [found] == record["file_matching_totalFileSize"]
    # the proof that the rule holds: the header of exactly that file says what the field says
    headers = record["file_header_request"]["general_architecture_by_file"]
    assert headers[found] == record["model_gguf_field"]["architecture"]


def test_the_field_is_not_a_statement_about_the_whole_repository():
    """One recorded repository holds files of another architecture (an importance matrix file): a
    repository-wide reading of the field would be wrong for them."""
    differing = [
        record["repo"]
        for record in _hub_records()
        if len({value for value in record["file_header_request"]["general_architecture_by_file"].values() if value}) > 1
    ]
    assert differing, "no recorded repository shows two architectures; the proof of the scope is gone"


@pytest.mark.parametrize(
    ("files", "total"),
    [
        ([("a.gguf", 5), ("b.gguf", 5)], 5),  # two files of that size: not one file
        ([("a.gguf", 5), ("b.gguf", 6)], 7),  # no file of that size
        ([("a.gguf", 5)], None),
        ([("a.gguf", 5)], True),
        ([("a.gguf", 5)], "5"),
        ([("a.gguf", 5)], 5.0),
        ([], 5),
    ],
)
def test_a_scope_that_cannot_be_shown_is_no_file(files, total):
    assert F.hub_metadata_file(files, total) is None


# --- V1/V2: the finding of one package -----------------------------------------------------------------------


def test_a_package_check_is_bound_to_repository_revision_file_architecture_source_commit_and_day():
    check = F.PackageCheck.model_validate(_check())
    assert (check.repo, check.revision, check.identity, check.file) == ("packager/Nova-9B-GGUF", REVISION, "Nova-9B-Q8_0.gguf", "Nova-9B-Q8_0.gguf")
    assert (check.architecture, check.result, check.llamacpp_commit, check.checked) == ("nova9", "absent", COMMIT, DAY)


def test_a_shard_set_is_one_package_read_at_its_first_shard():
    ok = _check(identity="Q8_0/Nova-9B-Q8_0.gguf", file="Q8_0/Nova-9B-Q8_0-00001-of-00002.gguf")
    assert F.PackageCheck.model_validate(ok).quantization == "Q8_0"
    for second in ("Q8_0/Nova-9B-Q8_0-00002-of-00002.gguf",):
        with pytest.raises(ValidationError, match="first shard"):
            F.PackageCheck.model_validate(_check(identity="Q8_0/Nova-9B-Q8_0.gguf", file=second))


FAULTY_CHECKS = {
    "revision of a branch": {"revision": "main"},
    "short revision": {"revision": "a" * 39},
    "upper-case revision": {"revision": "A" * 40},
    "commit of a branch": {"llamacpp_commit": "master"},
    "source of another commit": {"source_url": F.llamacpp_page_url("c" * 40)},
    "source of another repository": {"source_url": f"https://github.com/other/llama.cpp/blob/{COMMIT}/src/llama-arch.cpp"},
    "identity of another file": {"identity": "Nova-9B-Q4_K_M.gguf"},
    "file outside the repository": {"file": "../Nova-9B-Q8_0.gguf", "identity": "../Nova-9B-Q8_0.gguf"},
    "file with a drive": {"file": "/Nova-9B-Q8_0.gguf", "identity": "/Nova-9B-Q8_0.gguf"},
    "not a gguf file": {"file": "Nova-9B-Q8_0.bin", "identity": "Nova-9B-Q8_0.bin"},
    "quantization the file does not say": {"quantization": "Q4_K_M"},
    "architecture with a space": {"architecture": "nova 9"},
    "architecture in capitals": {"architecture": "NOVA9"},
    "architecture empty": {"architecture": ""},
    "the sentinel is no architecture": {"architecture": "(unknown)"},
    "a third result": {"result": "unknown"},
    "a third source": {"architecture_source": "model_card"},
    "an extra field": {"note": "x"},
    "repository without an owner": {"repo": "Nova-9B-GGUF"},
}


@pytest.mark.parametrize("name", sorted(FAULTY_CHECKS))
def test_a_faulty_package_check_is_refused(name):
    with pytest.raises(ValidationError):
        F.PackageCheck.model_validate(_check(**FAULTY_CHECKS[name]))


def test_a_package_check_has_only_two_results_because_unknown_is_the_absence_of_a_check():
    assert F.PACKAGE_RESULTS == ("present", "absent")
    assert F.ARCHITECTURE_SOURCES == ("hub_field", "file_header")


def test_the_hub_field_as_source_is_accepted_for_the_file_it_describes():
    assert F.PackageCheck.model_validate(_check(architecture_source="hub_field")).architecture_source == "hub_field"


def test_a_fresh_check_replaces_the_check_of_its_package_and_every_other_stays():
    old_a = F.PackageCheck.model_validate(_check(result="absent", checked=date(2026, 10, 1)))
    old_b = F.PackageCheck.model_validate(_check(identity="Nova-9B-Q4_K_M.gguf", file="Nova-9B-Q4_K_M.gguf", quantization="Q4_K_M", result="absent"))
    new_a = F.PackageCheck.model_validate(_check(result="present", checked=DAY, revision=OTHER_REVISION))
    assert F.merge_package_checks([old_a, old_b], [new_a]) == [new_a, old_b]


def test_nothing_fresh_changes_nothing_so_a_failed_run_removes_no_check():
    old = [F.PackageCheck.model_validate(_check())]
    assert F.merge_package_checks(old, []) == old


def test_an_older_check_never_replaces_a_newer_one():
    newer = F.PackageCheck.model_validate(_check(checked=DAY))
    older = F.PackageCheck.model_validate(_check(checked=date(2026, 10, 1), result="present"))
    assert F.merge_package_checks([newer], [older]) == [newer]


def test_the_merge_keeps_the_order_of_the_catalog_and_appends_new_packages():
    first = F.PackageCheck.model_validate(_check())
    second = F.PackageCheck.model_validate(_check(identity="Nova-9B-Q4_K_M.gguf", file="Nova-9B-Q4_K_M.gguf", quantization="Q4_K_M"))
    third = F.PackageCheck.model_validate(_check(repo="other/Nova-9B-GGUF"))
    assert F.merge_package_checks([first, second], [third]) == [first, second, third]


def test_the_merge_takes_a_repository_id_in_another_case_as_the_same_repository():
    """Hugging Face repository ids are case-insensitive; the file name and the revision are not touched."""
    old = F.PackageCheck.model_validate(_check(checked=date(2026, 10, 1)))
    fresh = F.PackageCheck.model_validate(_check(repo="Packager/nova-9b-gguf", result="present"))
    assert F.merge_package_checks([old], [fresh]) == [fresh]


# --- V7: the setup note of a model card ------------------------------------------------------------------------


def test_a_setup_note_names_the_model_the_card_revision_the_address_the_day_and_the_status():
    note = F.SetupNote.model_validate(_note())
    assert (note.model, note.status, note.card_revision, note.checked, note.package) == ("acme/Nova-9B", "instructions", REVISION, DAY, None)
    assert F.SETUP_STATUSES == ("instructions", "none_recognized", "not_read")


def test_a_package_name_belongs_to_a_note_that_found_instructions_and_nothing_else():
    assert F.SetupNote.model_validate(_note(package="nova-inference")).package == "nova-inference"
    with pytest.raises(ValidationError, match="package"):
        F.SetupNote.model_validate(_note(status="none_recognized", package="nova-inference"))


def test_a_card_that_was_not_read_has_no_revision_but_a_reason():
    unread = _note(status="not_read", card_revision=None, source_url="https://huggingface.co/acme/Nova-9B", reason="HTTP 503")
    assert F.SetupNote.model_validate(unread).reason == "HTTP 503"
    for bad in (dict(unread, card_revision=REVISION), dict(unread, reason=None), _note(reason="x")):
        with pytest.raises(ValidationError):
            F.SetupNote.model_validate(bad)


FAULTY_NOTES = {
    "address of another model": {"source_url": f"https://huggingface.co/other/Nova-9B/blob/{REVISION}/README.md"},
    "address of another revision": {"source_url": f"https://huggingface.co/acme/Nova-9B/blob/{OTHER_REVISION}/README.md"},
    "address of a branch": {"source_url": "https://huggingface.co/acme/Nova-9B/blob/main/README.md"},
    "revision of a branch": {"card_revision": "main"},
    "package with a space": {"package": "nova inference"},
    "package with an option": {"package": "-e nova"},
    "package with a version": {"package": "nova>=1"},
    "package too long": {"package": "n" * 65},
    "unknown status": {"status": "none"},
    "model that is no repository": {"model": "Nova"},
}


@pytest.mark.parametrize("name", sorted(FAULTY_NOTES))
def test_a_faulty_setup_note_is_refused(name):
    with pytest.raises(ValidationError):
        F.SetupNote.model_validate(_note(**FAULTY_NOTES[name]))


def test_a_reason_is_one_printable_line_of_at_most_ninety_seven_characters():
    base = dict(status="not_read", card_revision=None, source_url="https://huggingface.co/acme/Nova-9B")
    assert F.SetupNote.model_validate(_note(**base, reason="r" * 97))
    for bad in ("r" * 98, "two\nlines", "café"):
        with pytest.raises(ValidationError):
            F.SetupNote.model_validate(_note(**base, reason=bad))


def test_a_card_that_could_not_be_read_removes_no_note_and_adds_none():
    kept = F.SetupNote.model_validate(_note(package="nova-inference"))
    unread = F.SetupNote.model_validate(_note(status="not_read", card_revision=None, source_url="https://huggingface.co/acme/Nova-9B", reason="HTTP 503"))
    assert F.merge_setup_note(kept, unread) == kept
    assert F.merge_setup_note(None, unread) is None


def test_a_card_that_was_read_replaces_the_note_whatever_it_found():
    kept = F.SetupNote.model_validate(_note(package="nova-inference"))
    none_found = F.SetupNote.model_validate(_note(status="none_recognized", card_revision=OTHER_REVISION, source_url=f"https://huggingface.co/acme/Nova-9B/blob/{OTHER_REVISION}/README.md"))
    assert F.merge_setup_note(kept, none_found) == none_found
    assert F.merge_setup_note(None, none_found) == none_found


def test_the_limits_of_the_card_read_and_of_a_run_of_checks_are_fixed():
    assert F.CARD_MAX_BYTES == 128 * 1024
    assert (F.PACKAGE_ROWS_PER_RUN, F.PACKAGE_STAGE_BUDGET) == (40, 200)


# --- V4: who is the publisher, and what is recommended automatically -----------------------------------------


def test_the_publisher_is_the_owner_of_this_base_model_and_nobody_else():
    from modelroom.origin import package_origin

    assert package_origin("acme", "acme/Nova-9B", ["bartowski"]) == "publisher"
    assert package_origin("ACME", "acme/Nova-9B", []) == "publisher"  # an account name has no case on the Hub
    assert package_origin("bartowski", "acme/Nova-9B", ["bartowski"]) == "listed packager"
    assert package_origin("zeta", "acme/Nova-9B", ["bartowski"]) == "other"


def test_an_account_that_publishes_other_models_gets_no_release_for_a_foreign_conversion():
    """`search.owner_class` counts every publisher account of the catalog as `publisher`; here
    `zeta` owns another base model and converts one of `acme`."""
    from modelroom.catalog import load_catalog
    from modelroom.origin import listed_packagers_of, package_origin
    from modelroom.search import owner_class

    catalog = load_catalog()
    foreign = next(family.publisher for family in catalog.families if family.publisher != "acme")
    assert owner_class(foreign, catalog=catalog) == "publisher"  # the search class: any publisher account
    assert package_origin(foreign, "acme/Nova-9B", listed_packagers_of(catalog)) == "other"


def test_the_publisher_stands_over_the_listed_packager_when_one_account_is_both():
    from modelroom.origin import package_origin

    assert package_origin("acme", "acme/Nova-9B", ["acme"]) == "publisher"


def test_the_list_of_packagers_is_compared_without_case_on_both_sides():
    from modelroom.origin import package_origin

    assert package_origin("CONVERTER", "acme/Nova-9B", ["converter"]) == "listed packager"
    assert package_origin("converter", "acme/Nova-9B", ["CONVERTER"]) == "listed packager"
    assert package_origin("stranger", "acme/Nova-9B", ["CONVERTER"]) == "other"


def test_a_check_may_state_how_many_packages_of_its_quantization_the_repository_held():
    assert F.PackageCheck.model_validate(_check()).same_quantization is None
    assert F.PackageCheck.model_validate(_check(same_quantization=1)).same_quantization == 1
    with pytest.raises(ValidationError):
        F.PackageCheck.model_validate(_check(same_quantization=0))


def test_rows_without_any_check_are_taken_in_turn_over_several_runs_and_no_check_moves():
    from datetime import timedelta

    rows = [f"acme/Nova-{i:03d}" for i in range(100)]
    attempts: dict[str, date] = {}
    held = [F.PackageCheck.model_validate(_check())]
    seen: list[str] = []
    for day in range(3):
        today = DAY + timedelta(days=day)
        taken = F.package_rows_for_run(rows, attempts, today)
        assert len(taken) <= F.PACKAGE_ROWS_PER_RUN
        if day < 2:
            assert not set(taken) & set(seen)  # nobody twice before all were taken
        seen += taken
        attempts = F.record_attempts(attempts, taken, today)  # an empty or failed read counts as the attempt
        assert F.merge_package_checks(held, []) == held
    assert set(seen) == set(rows)
    assert not set(F.package_rows_for_run(rows, attempts, DAY + timedelta(days=2))) & set(seen[80:])  # taken today: not again
    again = F.package_rows_for_run(rows, attempts, DAY + timedelta(days=3))
    assert again == sorted(rows, key=lambda r: (attempts[r], r))[: F.PACKAGE_ROWS_PER_RUN]


def test_the_state_of_the_run_takes_an_id_in_another_case_as_the_same_row():
    """Codex b5: ids are case-insensitive; a row tried as `acme/Nova` is tried as `acme/nova` too. Variants already
    in the state are merged (the youngest day wins), and the ids handed out are the ones given."""
    from datetime import timedelta

    attempts = F.record_attempts({}, ["acme/Nova"], DAY)
    assert F.package_rows_for_run(["acme/nova", "acme/Other"], attempts, DAY) == ["acme/Other"]
    older, newer = DAY - timedelta(days=5), DAY - timedelta(days=1)
    mixed = {"ACME/nova": older, "acme/Nova": newer, "acme/Other": DAY - timedelta(days=3)}
    assert F.package_rows_for_run(["acme/nova", "acme/Other"], mixed, DAY) == ["acme/Other", "acme/nova"]
    merged = F.record_attempts(mixed, ["Acme/NOVA"], DAY)
    assert merged == {"Acme/NOVA": DAY, "acme/Other": DAY - timedelta(days=3)}
    assert F.record_attempts(mixed, [], DAY) == {"acme/Nova": newer, "acme/Other": DAY - timedelta(days=3)}


def test_the_listed_packagers_are_the_catalogs_own_list_and_the_shipped_default_where_it_has_none():
    from modelroom.catalog import Catalog
    from modelroom.origin import listed_packagers_of
    from modelroom.search import DEFAULT_PACKAGERS

    assert listed_packagers_of(Catalog(schema_version=2, packagers=["one", "two"])) == ("one", "two")
    assert listed_packagers_of(Catalog(schema_version=2)) == DEFAULT_PACKAGERS


def test_only_the_publisher_and_a_listed_packager_are_recommended_automatically():
    from modelroom.origin import RECOMMENDED_ORIGINS, auto_recommendable

    assert RECOMMENDED_ORIGINS == ("publisher", "listed packager")
    assert [auto_recommendable(origin) for origin in ("publisher", "listed packager", "other")] == [True, True, False]


# --- E3: the fixed wordings ----------------------------------------------------------------------------------


def test_the_wordings_of_the_decision_are_kept_to_the_letter():
    assert F.LOAD_BLOCKED == "{runtime} installation not offered for {model}: architecture {arch} is absent from the checked llama.cpp source (checked {date})"
    assert F.COMMUNITY_CONVERSION == "{packager} is a community conversion, not checked; not offered as the automatic choice"
    assert F.KNOWLEDGE_HINT == "{model} ({publisher}) is in the catalog; no GGUF from its publisher or a listed packager is known here"
    assert F.KNOWLEDGE_HINT_TENSOR == "tensor packages are shown when a machine with vLLM is selected; check the model card for runtime requirements"
    assert F.SETUP_NOTE == "{model} has setup instructions in its model card: {url}"
    assert F.SETUP_NOTE_PACKAGE == "it names the package {package} as required; the commands below are templates without it"


def test_the_load_block_says_what_was_checked_and_when_and_claims_nothing_more():
    check = F.PackageCheck.model_validate(_check())
    assert F.load_blocked_text("Ollama", "acme/Nova-9B", check) == (
        "Ollama installation not offered for acme/Nova-9B: architecture nova9 is absent from the checked llama.cpp source (checked 2026-10-13)"
    )
    with pytest.raises(ValueError, match="absent"):
        F.load_blocked_text("Ollama", "acme/Nova-9B", F.PackageCheck.model_validate(_check(result="present")))


def test_the_community_conversion_line_names_the_packager():
    assert F.community_conversion_text("othercorp") == "othercorp is a community conversion, not checked; not offered as the automatic choice"


def test_the_knowledge_hint_adds_the_tensor_sentence_only_on_a_positive_proof():
    plain = F.knowledge_hint_text("acme/Nova-9B", "acme", tensor_offer=False)
    assert plain == "acme/Nova-9B (acme) is in the catalog; no GGUF from its publisher or a listed packager is known here"
    assert F.knowledge_hint_text("acme/Nova-9B", "acme", tensor_offer=True) == (
        f"{plain}; tensor packages are shown when a machine with vLLM is selected; check the model card for runtime requirements"
    )


def test_the_setup_note_lines_are_one_and_two_only_for_a_named_package():
    plain = F.SetupNote.model_validate(_note())
    named = F.SetupNote.model_validate(_note(package="nova-inference"))
    assert F.setup_note_lines(plain) == (f"acme/Nova-9B has setup instructions in its model card: https://huggingface.co/acme/Nova-9B/blob/{REVISION}/README.md",)
    assert F.setup_note_lines(named)[1] == "it names the package nova-inference as required; the commands below are templates without it"
    assert len(F.setup_note_lines(named)) == 2
    for text in (_note(status="none_recognized"), _note(status="not_read", card_revision=None, source_url="https://huggingface.co/acme/Nova-9B", reason="HTTP 503")):
        assert F.setup_note_lines(F.SetupNote.model_validate(text)) == ()


def test_the_line_for_no_recommendation_left_is_fixed():
    assert F.NO_RECOMMENDATION == "no automatic recommendation: every row shown is blocked or not checked; choose a row by name"


def test_the_knowledge_hint_limit_and_the_direct_find_limit_are_fixed():
    assert F.KNOWLEDGE_HINT_MAX == 3


def test_the_forms_are_closed_models():
    for model in (F.PackageCheck, F.SetupNote):
        assert model.model_config["extra"] == "forbid"
    assert copy.deepcopy(F.PackageCheck.model_validate(_check())).model_dump(mode="json")["checked"] == "2026-10-13"
