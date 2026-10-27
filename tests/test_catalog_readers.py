"""The register of the tests that read the shipped catalog (`modelroom/catalog.json`), held by a test.

Every publication of the catalog moves `modelroom/catalog.json`. A test that reads it -- `load_catalog()` without a
path, or `CATALOG_FILE` -- can turn red on a publication that is right. Which tests those are stood nowhere; this
file is the list, and a test that compares it with the tests as they are.

Each file carries one word for why it reads the shipped catalog:

- `invariant` -- it checks the shipped catalog on purpose (its rows, its notes, its longest name). After a
  publication these, and only these, may turn red, and then for a reason that is the catalog's, not the test's.
- `source` -- it takes the shipped catalog as a handy source of data; the decoupling (a fixture catalog handed in with
  `shipped=`, `fixtures/catalog_guided.json`) is open. A series that is new here says so with this word, or does not read
  the file at all.

Whoever adds a reader, or lets one go, changes the list below in the same commit. The check looks at the text of
`tests/test_*.py` and `tests/fixture_support.py`; the root it looks under is a parameter, so a test can point it at a folder
of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The two forms of a read of the file of the package: the call without a path, and the constant itself.
_READ = re.compile(r"load_catalog\(\s*\)|\bCATALOG_FILE\b")

READERS = {
    "fixture_support.py": "source",
    "test_catalog.py": "invariant",
    "test_catalog_remote.py": "source",
    "test_catalog_task.py": "invariant",
    "test_guided.py": "source",
    "test_guided_acceptance.py": "source",
    "test_guided_folder.py": "source",
    "test_guided_install.py": "source",
    "test_guided_models.py": "source",
    "test_guided_screen.py": "invariant",  # the longest name of the shipped catalog (the Goldens read the fixture)
    "test_guided_search.py": "source",
    "test_guided_vllm_match.py": "source",
    "test_guided_write.py": "source",
    "test_install_vllm.py": "source",
    "test_ko_catalog_v3.py": "invariant",  # Z.107-112 hold the schema_version of the shipped file against its content; Z.210 also takes it as a source
    "test_ko_findings.py": "source",
    "test_ko_p5_channel.py": "source",
    "test_ko_render.py": "source",
    "test_pull_choice.py": "source",
    "test_pull_other.py": "source",
    "test_search.py": "source",
    "test_search_apply.py": "source",
    "test_search_pages.py": "source",
    "test_search_release.py": "invariant",
    "test_search_word.py": "source",
    "test_vs_switches.py": "source",
}
WORDS = {"invariant", "source"}


def shipped_catalog_readers(root: Path = HERE) -> set[str]:
    """The names of the files under `root` (`test_*.py` and `fixture_support.py`, this file excepted) that read the shipped catalog."""
    names = {path.name for path in root.glob("test_*.py")} | {"fixture_support.py"}
    names.discard(Path(__file__).name)
    return {
        name
        for name in names
        if (root / name).is_file() and _READ.search((root / name).read_text(encoding="utf-8"))
    }


def register_problems(root: Path = HERE) -> list[str]:
    """What is wrong with the list above when held against the tests under `root`; empty when it is right."""
    found = shipped_catalog_readers(root)
    problems = [f"{name}: reads the shipped catalog and is not listed: add it with its reason" for name in sorted(found - set(READERS))]
    problems += [f"{name}: listed, but no longer reads the shipped catalog: take it off the list" for name in sorted(set(READERS) - found)]
    return problems


def test_the_tests_that_read_the_shipped_catalog_are_the_listed_ones():
    assert register_problems() == []


def test_every_listed_reader_names_its_reason_in_one_of_the_two_words():
    assert set(READERS.values()) <= WORDS


def test_a_test_the_list_does_not_know_is_found_under_another_root(tmp_path):
    (tmp_path / "test_newcomer.py").write_text("from modelroom.catalog import load_catalog\n\ncatalog = load_catalog()\n", encoding="utf-8")
    (tmp_path / "test_quiet.py").write_text("from modelroom.catalog import load_catalog\n\ncatalog = load_catalog(PATH)\n", encoding="utf-8")

    assert shipped_catalog_readers(tmp_path) == {"test_newcomer.py"}
    problems = register_problems(tmp_path)  # what the first test would refuse, were the root the tests' own
    assert "test_newcomer.py: reads the shipped catalog and is not listed: add it with its reason" in problems
