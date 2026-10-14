"""AP9-KO P1, part 3: `modelroom pull` and the stored check (V10, decision A of the management, 2026-10-13).

After the name is validated and before the daemon is touched, the command asks the catalog it reads offline
(`Catalog.pull_check`). A stored `absent` check for `hf.co/<repo>:<quant>` -- the name carries no revision, so the check
decides on the content last checked, and its reason names the day -- ends the command with exit 2, `--yes` included.
An Ollama name, a name the catalog does not lead to one check for, and a check that is not `absent` take the old way.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modelroom.catalog_findings import PackageCheck, load_blocked_text
from modelroom.cli import main
from modelroom.cli_pull import NO_TERMINAL, run_pull
from modelroom.daemon import PULL_PATH

from fixture_room import fixed_room
from test_cli_pull import _daemon
from test_ko_render import _catalog, _check_for
from test_render import _hf_package

NOW = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
NAME = "hf.co/packager/Nova-8B-GGUF:Q4_K_M"
MODEL = "acme/Nova-8B"


def _blocking(**changes) -> dict:
    return _check_for(_hf_package(), same_quantization=1, **changes)


def _reason(check: dict) -> str:
    return load_blocked_text("Ollama", MODEL, PackageCheck.model_validate(check))


def test_a_stored_absent_check_ends_the_command_with_exit_2_before_the_daemon_and_yes_does_not_pass_it(capsys):
    check = _blocking()
    catalog = _catalog([check])

    for yes in (False, True):
        daemon = _daemon()
        code = run_pull(NAME, weights_bytes=1024, yes=yes, daemon=daemon, room=fixed_room(), now=NOW, catalog=catalog)

        assert code == 2
        assert capsys.readouterr().err.strip() == _reason(check)  # the words of the table, with the day of the check
        assert daemon.calls == []  # the daemon was never asked, not even for its version
    assert "2026-10-13" in _reason(check)


def test_the_name_carries_no_revision_so_a_check_of_another_revision_still_blocks_the_movable_name(capsys):
    """Variant A: a decision on the content last checked, not a statement about what the name serves now."""
    check = _blocking(revision="b" * 40)  # the package the snapshot knows has revision `a...`; the name has none

    assert run_pull(NAME, yes=True, daemon=_daemon(), room=fixed_room(), now=NOW, catalog=_catalog([check])) == 2
    assert capsys.readouterr().err.strip() == _reason(check)


def test_the_lookup_is_by_repository_and_quantization_without_case():
    check = _blocking()

    assert run_pull("hf.co/PACKAGER/nova-8b-gguf:q4_k_m", yes=True, daemon=_daemon(), room=fixed_room(), now=NOW, catalog=_catalog([check])) == 2


@pytest.mark.parametrize(
    "name, check",
    [
        ("hf.co/packager/Nova-8B-GGUF:Q8_0", _blocking()),  # another quantization of the repository
        ("hf.co/packager/Other-GGUF:Q4_K_M", _blocking()),  # another repository
        ("granite4.2:8b", _blocking()),  # an Ollama name inherits no finding of the Hub through its base model
        (NAME, _blocking(result="present")),  # only `absent` blocks
        (NAME, {k: v for k, v in _blocking().items() if k != "same_quantization"}),  # a second package nobody checked could hide
    ],
    ids=["other-quant", "other-repo", "ollama-name", "present", "not-unique"],
)
def test_every_other_name_takes_the_way_it_always_took(name, check, capsys):
    daemon = _daemon()

    code = run_pull(name, weights_bytes=1024, yes=False, daemon=daemon, room=fixed_room(), now=NOW, catalog=_catalog([check]))

    assert code == 2 and NO_TERMINAL in capsys.readouterr().err  # it reached the question (no terminal under pytest)
    assert daemon.calls  # and the daemon was asked


def test_without_a_catalog_the_command_is_what_it_was(capsys):
    daemon = _daemon()

    code = run_pull(NAME, weights_bytes=1024, yes=False, daemon=daemon, room=fixed_room(), now=NOW)

    assert code == 2 and NO_TERMINAL in capsys.readouterr().err and daemon.calls


def test_the_two_machines_of_a_handover_may_hold_different_catalogs_the_one_that_pulls_decides(capsys):
    daemon = _daemon()
    # the machine that wrote the handover knew the check; this one holds a catalog without it
    code = run_pull(NAME, weights_bytes=1024, yes=False, daemon=daemon, room=fixed_room(), now=NOW, catalog=_catalog([]))

    assert code == 2 and NO_TERMINAL in capsys.readouterr().err and daemon.calls


def test_the_command_line_hands_the_catalog_it_reads_to_the_pull(tmp_path, capsys):
    check = _blocking()
    daemon = _daemon()

    code = main(
        ["pull", NAME, "--yes"], now=NOW, pointer_path=tmp_path / "guided.json", daemon=daemon, room=fixed_room(),
        catalog=_catalog([check]),
    )

    assert code == 2 and capsys.readouterr().err.strip() == _reason(check)
    assert daemon.calls == [] and not any(call[1] == PULL_PATH for call in daemon.calls)


def test_the_command_line_without_a_given_catalog_reads_one_offline_and_asks_the_network_nothing(tmp_path, capsys):
    """The shipped catalog holds no check: the name takes the old way (and the daemon is asked)."""
    daemon = _daemon()

    code = main(["pull", NAME], now=NOW, pointer_path=tmp_path / "guided.json", daemon=daemon, room=fixed_room())

    assert code == 2 and NO_TERMINAL in capsys.readouterr().err and daemon.calls
