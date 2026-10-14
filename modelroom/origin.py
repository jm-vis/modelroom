"""Whose conversion a GGUF package is, for the one decision it steers: the automatic installation (AP9-KO, V4).

Three origins, by the owner of the repository against the base model it converts:

- `publisher`: the owner of **this** base model. An account that publishes other models of the catalog gets
  no release for a conversion of a model it does not own, which is why `search.owner_class` (any publisher
  account of the catalog) is not used here;
- `listed packager`: an account of the catalog's own list of packagers (`Catalog.packagers`; the shipped default
  where a catalog has none -- no list of its own is kept for this rule, and a list a folder's configuration
  adds changes what is searched, not what is recommended);
- `other`.

The publisher stands over the listed packager when one account is both. The origin steers only the
automatic recommendation (`auto_recommendable`); it is no rating of a package, no fit and no ranking.
"""

from __future__ import annotations

from typing import Literal, Sequence

from .catalog import Catalog
from .search import DEFAULT_PACKAGERS

Origin = Literal["publisher", "listed packager", "other"]
RECOMMENDED_ORIGINS: tuple[Origin, ...] = ("publisher", "listed packager")


def listed_packagers_of(catalog: Catalog) -> tuple[str, ...]:
    """The accounts the automatic recommendation trusts as packagers: the catalog's list, else the shipped default."""
    return tuple(catalog.packagers) if catalog.packagers else tuple(DEFAULT_PACKAGERS)


def package_origin(owner: str, base_model: str, packagers: Sequence[str]) -> Origin:
    """The origin of a repository of account `owner` that converts `base_model` (`<account>/<name>`)."""
    if owner.casefold() == base_model.split("/", 1)[0].casefold():
        return "publisher"
    return "listed packager" if owner.casefold() in {name.casefold() for name in packagers} else "other"


def auto_recommendable(origin: Origin) -> bool:
    """Whether a package of this origin may be the automatic recommendation; an explicit choice never needs it."""
    return origin in RECOMMENDED_ORIGINS
