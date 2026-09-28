"""The folder question of step 1 (AP9-UX2 U1, U7; GF F1, F2): a remembered folder, a home folder.

The home folder of these tests is the one `tests/conftest.py` puts in `HOME`/`USERPROFILE`, so
`Path.home()` is a folder of the test. Nothing outside `tmp_path` and that folder is written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.binding import POINTER_SCHEMA_VERSION, GuidedPointer, read_pointer, write_pointer
from modelroom.dialog import AnswerInvalidError, AnswerMissingError, FileAsker
from modelroom.guided_folder import CONFIG_NAME, HOME_AGAIN, HOME_NOTE, ROOT_NOTE, config_file

NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
CONFIG = 'schema_version = 3\npackagers = []\npublishers = []\n\n[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n'


class _Asker:
    """An asker like a terminal, not an answer file: every key answers from a queue of its own, so the
    same key may answer differently the next time it is asked (`guided_folder` tells the two apart
    by `isinstance(asker, FileAsker)`)."""

    def __init__(self, answers: dict[str, list]) -> None:
        self._queues = {key: list(values) for key, values in answers.items()}
        self.asked: list[str] = []
        self.lists: dict[str, list] = {}

    def _answer(self, key, question):
        self.asked.append(key)
        if not self._queues.get(key):
            raise AnswerMissingError(key, question)
        return self._queues[key].pop(0)

    def select(self, key, question, choices, instruction=None):
        self.lists[key] = list(choices)
        return self._answer(key, question)

    def text(self, key, question, default=""):
        return str(self._answer(key, question))

    def confirm(self, key, question, default=True):
        return bool(self._answer(key, question))


_Terminal = _Asker


def _run(tmp_path: Path, asker, here: Path):
    from fixture_room import fixed_room
    from fixture_support import offline_servers, windows_probes
    from modelroom.catalog import load_catalog
    from modelroom.guided import GuidedRun
    from modelroom.http import FixtureTransport
    from modelroom.local_servers import ServerRun
    from modelroom.screen import Screen

    lines: list[str] = []
    run = GuidedRun(
        asker=asker,
        here=here,
        pointer_path=tmp_path / "pointer" / "guided.json",
        transport=FixtureTransport({}),
        probes=windows_probes(),
        now=NOW,
        catalog=load_catalog(),
        servers=ServerRun(offline_servers()),
        out=lines.append,
        screen=Screen(lines.append, colored=False),
        room=fixed_room(),
    )
    return run, lines


def _remember(tmp_path: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / CONFIG_NAME).write_bytes(CONFIG.encode("utf-8"))
    pointer = GuidedPointer(schema_version=POINTER_SCHEMA_VERSION, current=str(folder.resolve()), bindings={})
    write_pointer(tmp_path / "pointer" / "guided.json", pointer)


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


# --- U1: a remembered folder that is not this one ------------------------------------------------------


def test_a_remembered_folder_that_is_not_this_one_is_a_question(tmp_path: Path):
    last, here = tmp_path / "last", tmp_path / "here"
    here.mkdir()
    _remember(tmp_path, last)
    asker = _Asker({"results": ["last"]})
    run, lines = _run(tmp_path, asker, here)

    chosen = config_file(run, None)

    assert "results" in asker.asked
    assert chosen == (last / CONFIG_NAME).resolve()
    assert "? Where should results live?  the folder of the last run" in [line.strip() for line in lines]


def test_last_is_under_the_pointer(tmp_path: Path):
    last, here = tmp_path / "last", tmp_path / "here"
    here.mkdir()
    _remember(tmp_path, last)
    asker = _Asker({"results": ["here"]})
    run, _lines = _run(tmp_path, asker, here)

    config_file(run, None)

    choices = asker.lists["results"]
    assert [choice.value for choice in choices] == ["last", "here", "path"]
    assert choices[0].checked and not any(choice.checked for choice in choices[1:])
    assert choices[0].label == f"the folder of the last run ({last.resolve()})"
    assert choices[1].label == f"this folder ({here.resolve()})"


def test_the_same_folder_asks_nothing(tmp_path: Path):
    here = tmp_path / "here"
    _remember(tmp_path, here)
    asker = _Asker({})
    run, _lines = _run(tmp_path, asker, here)

    assert config_file(run, None) == (here / CONFIG_NAME).resolve() and asker.asked == []


def test_here_takes_this_folder_and_remembers_it(tmp_path: Path):
    from modelroom.guided import _remember_folder

    last, here = tmp_path / "last", tmp_path / "here"
    here.mkdir()
    _remember(tmp_path, last)
    run, _lines = _run(tmp_path, _Asker({"results": ["here"]}), here)

    chosen = config_file(run, None)
    _remember_folder(run, chosen.parent)

    assert chosen == (here / CONFIG_NAME).resolve()
    assert read_pointer(run.pointer_path).current == str(here.resolve())


# --- U7: a home folder or the top of a drive ---------------------------------------------------------


def test_the_home_folder_is_a_question(tmp_path: Path):
    home = Path.home().resolve()
    asker = _Asker({"results": ["here"], "write_home": [True]})
    run, lines = _run(tmp_path, asker, home)

    assert config_file(run, None) == home / CONFIG_NAME
    assert HOME_NOTE in [line.strip() for line in lines]
    assert "? Write the results into this folder anyway?  Yes" in [line.strip() for line in lines]


def test_the_top_of_a_drive_is_a_question(tmp_path: Path):
    root = Path(tmp_path.anchor)
    elsewhere = tmp_path / "results"
    asker = _Asker({"results": ["here"], "write_home": [False], "results_path": [str(elsewhere)]})
    run, lines = _run(tmp_path, asker, root)

    assert config_file(run, None) == elsewhere.resolve() / CONFIG_NAME
    assert ROOT_NOTE in [line.strip() for line in lines]


def test_no_to_the_home_folder_offers_the_modelroom_folder_in_it(tmp_path: Path):
    """F2: the path question offers `<home>/modelroom`."""
    home = Path.home().resolve()
    offered: list[str] = []

    class _Offer(_Asker):
        def text(self, key, question, default=""):
            offered.append(default)
            return super().text(key, question, default)

    run, _lines = _run(tmp_path, _Offer({"results": ["here"], "write_home": [False], "results_path": [str(tmp_path / "r")]}), home)
    config_file(run, None)

    assert offered == [str(home / "modelroom")]


def test_no_to_the_home_folder_writes_and_migrates_nothing(tmp_path: Path):
    """A home folder with a configuration of an earlier schema: after No, nothing there is touched."""
    from fixture_support import FIXTURES, build_transport, guided_transport_mapping, offline_daemon, offline_servers
    from fixture_support import windows_probes
    from modelroom.guided import run_guided

    home = Path.home().resolve()
    (home / CONFIG_NAME).write_bytes((FIXTURES / "config_v1" / "config.toml").read_bytes())
    before = _tree(home)
    elsewhere = tmp_path / "results"
    answers = {
        "results": "here", "write_home": False, "results_path": str(elsewhere), "machines": [],
        "search": "qwen", "filter_owners": True, "select": [], "users": 1, "context": "L",
    }

    run_guided(
        FileAsker(answers), here=home, pointer_path=tmp_path / "pointer" / "guided.json",
        transport=build_transport(guided_transport_mapping()), probes=windows_probes(), daemon=offline_daemon(),
        servers=offline_servers(), now=NOW, out=lambda line: None, colored=False,
    )

    assert _tree(home) == before
    assert (elsewhere / CONFIG_NAME).is_file()


def test_home_again_after_no_asks_again(tmp_path: Path):
    """At a terminal the question comes again as long as the answer is the home folder."""
    home = Path.home().resolve()
    elsewhere = tmp_path / "results"
    asker = _Terminal({"results": ["here"], "write_home": [False, False], "results_path": [str(home), str(elsewhere)]})
    run, _lines = _run(tmp_path, asker, home)

    assert config_file(run, None) == elsewhere.resolve() / CONFIG_NAME
    assert asker.asked.count("write_home") == 2


def test_an_answer_file_naming_home_again_ends_with_exit_2_and_writes_nothing(tmp_path: Path, capsys):
    from fixture_support import build_transport, guided_transport_mapping, offline_daemon, offline_servers, windows_probes
    from modelroom.cli import main

    home = Path.home().resolve()
    before = _tree(home)
    answers = {"results": "here", "write_home": False, "results_path": str(home)}

    code = main(
        [], transport=build_transport(guided_transport_mapping()), probes=windows_probes(),
        pointer_path=tmp_path / "pointer" / "guided.json", now=NOW, asker=FileAsker(answers), here=home,
        out=lambda line: None, daemon=offline_daemon(), servers=offline_servers(),
    )

    assert code == 2 and HOME_AGAIN in capsys.readouterr().err
    assert _tree(home) == before
    with pytest.raises(AnswerInvalidError):
        config_file(_run(tmp_path, FileAsker(answers), home)[0], None)


def test_an_answer_file_without_write_home_is_a_missing_answer(tmp_path: Path):
    run, _lines = _run(tmp_path, FileAsker({"results": "here"}), Path.home().resolve())
    with pytest.raises(AnswerMissingError):
        config_file(run, None)


# --- `--config` ------------------------------------------------------------------------------------


def test_an_existing_config_argument_asks_no_folder_question(tmp_path: Path):
    folder = tmp_path / "given"
    folder.mkdir()
    (folder / CONFIG_NAME).write_bytes(CONFIG.encode("utf-8"))
    asker = _Asker({})
    run, _lines = _run(tmp_path, asker, Path.home().resolve())

    assert config_file(run, folder / CONFIG_NAME) == (folder / CONFIG_NAME).resolve()
    assert asker.asked == []


def test_a_missing_config_argument_falls_back_and_checks_the_home_folder(tmp_path: Path):
    home = Path.home().resolve()
    asker = _Asker({"results": ["here"], "write_home": [False], "results_path": [str(tmp_path / "r")]})
    run, lines = _run(tmp_path, asker, home)

    chosen = config_file(run, tmp_path / "nowhere" / CONFIG_NAME)

    assert chosen == (tmp_path / "r").resolve() / CONFIG_NAME
    assert "write_home" in asker.asked and HOME_NOTE in [line.strip() for line in lines]
