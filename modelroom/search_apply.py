"""The resolution of one search, written back into a configuration.

Split out of `modelroom/search.py` on 2026-09-25, unchanged: the search grew the three forms of
input it reads and the file passed the house code-mass threshold. `search.py` re-exports
`apply_hits`, `family_name_for` and `write_configuration`, so every caller still reads them there.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .catalog import Catalog
from .config import BaseModelConfig, ConfigError, Configuration, config_from_text
from .guided_contracts import SearchHit
from .state import acquire_lock, atomic_write_text, release_lock
from .toml_writer import dump_toml


def apply_hits(config: Configuration, hits: Iterable[SearchHit], *, catalog: Catalog) -> Configuration:
    """`configuration + resolution -> new configuration`, pure: `config` is never changed.

    Every resolved hit adds (or reuses) its base model's family, lists the base model's account
    under `publishers`, and adds the hit's own repository as an owner-bound `repos` target of
    that base model -- the search finds repositories under accounts the packager list does not
    name, and `repos` is where such a target belongs (CONTRACTS.md, "BaseModelConfig").
    Unresolved hits are ignored: nothing unproven ever reaches the configuration. Repeating the
    same hits changes nothing. The result is validated like any other configuration, so a
    target two base models would claim raises `ConfigError` here rather than at fetch time.

    An Ollama name belongs to the **base model**, not to the repository that led to it, so it is
    collected per base model before anything is written: the name the hits carry wins over one
    the configuration already holds (the user has just decided it), and two hits of the same base
    model naming *different* Ollama packages are contradictory input, refused with `ConfigError`
    rather than settled by whichever hit came first.
    """
    resolved = [hit for hit in hits if hit.resolved and hit.resolved_base_model is not None]
    ollama_by_base = _ollama_by_base(resolved)

    data = config.model_dump(mode="json")
    for hit in resolved:
        base = str(hit.resolved_base_model)
        publisher = base.split("/", 1)[0]
        if publisher not in data["publishers"]:
            data["publishers"] = [*data["publishers"], publisher]
        family = _family_entry(data, base, catalog)
        base_entry = _base_model_entry(family, base)
        if hit.repo not in base_entry["repos"]:
            base_entry["repos"] = [*base_entry["repos"], hit.repo]
        name = ollama_by_base.get(base)
        if name is not None:
            base_entry["ollama_base"], _, base_entry["ollama_tag"] = name.partition(":")
    return Configuration.from_dict(data)


def _ollama_by_base(resolved: Sequence[SearchHit]) -> dict[str, str]:
    """The one Ollama name each base model's hits agree on; `ConfigError` when they disagree."""
    names: dict[str, set[str]] = {}
    for hit in resolved:
        if hit.ollama is not None:
            names.setdefault(str(hit.resolved_base_model), set()).add(hit.ollama)
    agreed: dict[str, str] = {}
    for base, candidates in names.items():
        if len(candidates) > 1:
            raise ConfigError(
                f"{base}: the selected repositories name different Ollama packages "
                f"({', '.join(sorted(candidates))}); a base model has one"
            )
        agreed[base] = candidates.pop()
    return agreed


def _family_entry(data: dict, base: str, catalog: Catalog) -> dict:
    """The family dict that holds `base`, created when no family lists it yet."""
    for family in data["families"]:
        if any(entry["hf_repo"] == base for entry in family["base_models"]):
            return family
    name = family_name_for(catalog, base)
    for family in data["families"]:
        if family["name"] == name:
            return family
    family: dict = {"name": name, "base_models": []}
    data["families"] = [*data["families"], family]
    return family


def _base_model_entry(family: dict, base: str) -> dict:
    """The base model dict inside `family`, created without an Ollama name when it is new.

    The Ollama name is set by `apply_hits` afterwards, once per base model, so it does not
    depend on which hit created the entry.
    """
    for entry in family["base_models"]:
        if entry["hf_repo"] == base:
            return entry
    entry = BaseModelConfig(hf_repo=base).model_dump(mode="json")
    family["base_models"].append(entry)
    return entry


def family_name_for(catalog: Catalog, hf_repo: str) -> str:
    """The family a base model belongs to: the catalog's name, else one made from the repo name.

    A made-up name is the repository's own name in the shape a family name has to have
    (`[a-z0-9][a-z0-9.-]*`): lower case, every other character a hyphen.
    """
    for family in catalog.families:
        if any(model.hf_repo == hf_repo for model in family.models):
            return family.name
    name = hf_repo.split("/", 1)[1].lower()
    slug = "".join(character if character.isalnum() or character in ".-" else "-" for character in name)
    slug = slug.lstrip(".-")
    if not slug:
        raise ValueError(f"cannot make a family name from {hf_repo!r}")
    return slug


_CONFIG_HEADER = (
    "modelroom configuration, written by the guided mode.\n"
    "Families and their owner-bound `repos` targets come from the search; everything else is\n"
    "kept as it was. Edit it by hand at any time -- the guided mode reads it back."
)


def write_configuration(path: Path, config: Configuration, *, now: datetime) -> None:
    """Write `config` to `path` as TOML, under the state lock, once it reads back unchanged.

    `path` is resolved once, and the resolved path is used for both the check and the write --
    the same thing `load_config` does when it reads the file back, so the directory that confines
    `paths.state` is the same one in all three places. A path that is a symlink therefore lands in
    the file it points at, not as a new file over the link.

    The text is produced and read back through `config.config_from_text`, the very reader
    `load_config` uses -- so the schema gate, the path resolution and both path confinement
    checks are the ones the next `modelroom` command will apply -- and the result is compared
    with `config`. All of that happens *before* the lock is taken and anything is written, so a
    configuration that would not read back never reaches the disk (the same order `modelroom
    migrate` uses). The lock is the same `modelroom.lock` every other writer takes: a `fetch`
    running in parallel makes this stop with `LockHeldError` instead of writing over the file it
    is reading, and it is released whether the write succeeds or fails. No backup is kept: the
    guided mode has the user confirm the change before it calls this.
    """
    target = path.resolve()
    text = dump_toml(config.model_dump(mode="json"), _CONFIG_HEADER)
    reread = config_from_text(text, target)
    if reread.model_dump(mode="json") != config.model_dump(mode="json"):
        raise ConfigError(f"{target}: the written configuration does not read back unchanged")

    config.paths.state.mkdir(parents=True, exist_ok=True)
    handle = acquire_lock(config.paths.lock_file, "search", now)
    try:
        atomic_write_text(target, text)
    finally:
        release_lock(handle)


__all__ = ["apply_hits", "family_name_for", "write_configuration"]
