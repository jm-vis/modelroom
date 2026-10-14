"""AP9-KO P5, part 1: the size scale of step 3 counts a package an `absent` check blocks as not fitting, so the column
agrees with step 5 (which carries such a package locked, or as `not_covered` where the fit sets it aside).

The check is read through the catalog the run uses (`Catalog.package_check`), the same lookup the render makes; a
`present` check, a check of another revision and no catalog count nothing differently.
"""

from __future__ import annotations

import json

from modelroom.guided_context import LEVELS, fit_texts

from test_guided_context import BY_REPO, PACKAGES, _checked, _profile
from test_ko_render import _catalog, _check_for


def _texts(catalog=None, packages=PACKAGES) -> dict[int, str]:
    return fit_texts(packages, BY_REPO, _checked(_profile()), [level.tokens for level in LEVELS], catalog=catalog)


def test_a_blocked_package_does_not_count_as_fitting_and_the_total_stays_what_is_shown():
    plain = _texts()
    blocked = _texts(_catalog([_check_for(PACKAGES[0])]))

    assert plain[8192] == "all 3 packages fit"
    assert blocked[8192] == "2 of 3 packages fit"


def test_every_package_blocked_is_none_of_them():
    catalog = _catalog([_check_for(package) for package in PACKAGES])

    assert _texts(catalog)[8192] == "none of 3 packages fits"


def test_a_check_that_does_not_block_and_no_catalog_change_nothing():
    plain = _texts()
    present = _catalog([_check_for(PACKAGES[0], result="present")])
    other_revision = _catalog([_check_for(PACKAGES[0], revision="b" * 40)])

    assert _texts(present) == plain
    assert _texts(other_revision) == plain
    assert _texts(None) == plain


def test_a_blocked_package_does_not_make_the_column_say_more_than_without_it():
    """Blocking can only take a package out of the count: no level shows more fitting packages than the plain column."""
    plain = _texts()
    blocked = _texts(_catalog([_check_for(PACKAGES[1])]))

    def fitting(text: str) -> int:
        return len(PACKAGES) if text.startswith("all") else 0 if text.startswith("none") else int(text.split(" ", 1)[0])

    assert all(fitting(blocked[level.tokens]) <= fitting(plain[level.tokens]) for level in LEVELS)
    assert json.dumps(plain) != json.dumps(blocked)
