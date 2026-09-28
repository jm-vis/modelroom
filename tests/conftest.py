"""Two rules for the whole suite: no test may reach the home folder, nor the room, of the person running it.

Every test already passes its own `pointer_path` into `tmp_path`, so nothing here reads or writes
`~/.modelroom` by design. This fixture is the guard behind that design rather than a second way of
doing it: it moves `HOME` and `USERPROFILE` into a folder of the test, so a call that forgets to
pass a path -- `binding.default_pointer_path()`, `Path.home()`, `Path.expanduser()` -- lands in
`tmp_path` and not in a real home folder. `scripts/selftest.py` does the same for its own run.

The room (AP9-UX2 U8, Codex r2-5): the free graphics memory and disk a guided run reads live come
from `fixture_room.fixed_room()` for every test; the real probes refuse to run under pytest
(`room.real_room_probes`), so a test that forgot this fails instead of reading the machine.
"""

from __future__ import annotations

import pytest

from fixture_room import fixed_room


@pytest.fixture(autouse=True)
def _home_in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture(autouse=True)
def _fixed_room(monkeypatch: pytest.MonkeyPatch):
    from modelroom import room

    monkeypatch.setattr(room, "default_room_probes", lambda: fixed_room())
