"""One rule for the whole suite: no test may reach the home folder of the person running it.

Every test already passes its own `pointer_path` into `tmp_path`, so nothing here reads or writes
`~/.modelroom` by design. This fixture is the guard behind that design rather than a second way of
doing it: it moves `HOME` and `USERPROFILE` into a folder of the test, so a call that forgets to
pass a path -- `binding.default_pointer_path()`, `Path.home()`, `Path.expanduser()` -- lands in
`tmp_path` and not in a real home folder. `scripts/selftest.py` does the same for its own run.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _home_in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home
