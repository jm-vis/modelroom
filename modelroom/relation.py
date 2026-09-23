"""The relation check: does a Hugging Face repo declare itself a quantization of one base model?

One pure function for search and fetch alike. It reads the repo's `tags` (where Hugging Face
publishes `base_model:<repo>` and `base_model:<relation>:<repo>`) and its `cardData`
(`base_model`, `base_model_relation`) and compares them with the configured `hf_repo`. A pass
(`quantized`) allows at most `metadata_ok` together with the other provenance rules; it never
makes a package `approved`, which only a human `Approval` does (CONTRACTS.md, "Relation check").
"""

from __future__ import annotations

from typing import Literal

RelationStatus = Literal["quantized", "derivative", "relation_unknown", "metadata_conflict", "base_model_tag"]

_DERIVATIVE_RELATIONS = frozenset({"finetune", "adapter", "merge"})
_TAG_PREFIX = "base_model:"


class _MalformedMetadata(ValueError):
    """A metadata field has a shape the Hub does not publish; never guessed around."""


def _tag_facts(tags: list[str] | None) -> tuple[set[str], set[tuple[str, str]]]:
    """Plain bases (`base_model:<repo>`) and relations (`base_model:<relation>:<repo>`) from `tags`.

    `tags` is a list or `None`; any other container raises `_MalformedMetadata`.
    """
    bases: set[str] = set()
    relations: set[tuple[str, str]] = set()
    if tags is None:
        return bases, relations
    if not isinstance(tags, list):
        raise _MalformedMetadata(f"tags is not a list: {tags!r}")
    for tag in tags:
        if not isinstance(tag, str):
            raise _MalformedMetadata(f"a tag is not a string: {tag!r}")
        if not tag.startswith(_TAG_PREFIX):
            continue
        rest = tag[len(_TAG_PREFIX) :]
        relation, separator, repo = rest.partition(":")
        if separator:
            relations.add((relation, repo))
        else:
            bases.add(rest)
    return bases, relations


def _card_facts(card_data: dict | None) -> tuple[set[str] | None, str | None]:
    """`cardData.base_model` as a set (`None` when absent) and `cardData.base_model_relation`.

    `card_data` is a dict or `None`, `base_model` a string or a list of strings,
    `base_model_relation` a string; any other shape raises `_MalformedMetadata`.
    """
    if card_data is None:
        return None, None
    if not isinstance(card_data, dict):
        raise _MalformedMetadata(f"cardData is not an object: {card_data!r}")
    raw_base = card_data.get("base_model")
    if raw_base is None:
        card_bases = None
    elif isinstance(raw_base, str):
        card_bases = {raw_base}
    elif isinstance(raw_base, list) and all(isinstance(entry, str) for entry in raw_base):
        card_bases = set(raw_base)
    else:
        raise _MalformedMetadata(f"cardData.base_model has an unexpected shape: {raw_base!r}")
    relation = card_data.get("base_model_relation")
    if relation is not None and not isinstance(relation, str):
        raise _MalformedMetadata(f"cardData.base_model_relation is not a string: {relation!r}")
    return card_bases, relation


def check_relation(tags: list[str] | None, card_data: dict | None, hf_repo: str) -> tuple[RelationStatus, str]:
    """Return `(status, reason)` for one repo against the configured base model `hf_repo`.

    - `metadata_conflict`: malformed metadata, tags and card name different bases, a relation
      tag names a repo that is not a declared base, or tags and card state different relations;
    - `base_model_tag`: not exactly one declared base, or the one base is not `hf_repo`;
    - `relation_unknown`: no relation declared, or one this check does not know;
    - `derivative`: `finetune`, `adapter` or `merge`;
    - `quantized`: exactly the one base `hf_repo` with relation `quantized` -- the only pass.
    """
    try:
        tag_bases, tag_relations = _tag_facts(tags)
        card_bases, card_relation = _card_facts(card_data)
    except _MalformedMetadata as exc:
        return "metadata_conflict", str(exc)

    if tag_bases and card_bases is not None and tag_bases != card_bases:
        return "metadata_conflict", f"tags name {sorted(tag_bases)}, cardData names {sorted(card_bases)}"
    bases = tag_bases | (card_bases or set())
    stray = sorted(repo for _, repo in tag_relations if repo not in bases)
    if stray:
        return "metadata_conflict", f"relation tag names {stray}, which is not a declared base"
    relations = {relation for relation, _ in tag_relations}
    if card_relation is not None:
        if relations and relations != {card_relation}:
            return "metadata_conflict", f"tags state {sorted(relations)}, cardData states {card_relation!r}"
        relations.add(card_relation)
    if len(relations) > 1:
        return "metadata_conflict", f"more than one relation declared: {sorted(relations)}"
    if bases != {hf_repo}:
        return "base_model_tag", f"declared bases {sorted(bases)}, expected exactly [{hf_repo!r}]"

    if not relations:
        return "relation_unknown", "no base_model relation declared"
    relation = relations.pop()
    if relation == "quantized":
        return "quantized", f"quantized from {hf_repo}"
    if relation in _DERIVATIVE_RELATIONS:
        return "derivative", f"{relation} of {hf_repo}, not a quantization"
    return "relation_unknown", f"unknown relation {relation!r}"
