"""Decide whether a package's link to its base model can be trusted.

A package's link holds either because a human approved its exact content (an approval
survives only as long as the content it names is still current) or because its own metadata
proves the link on its own: on Hugging Face, the repository has to be one of the base model's
package targets *and* declare itself a `quantized` build of exactly that base model
(`relation.check_relation`, the one relation check search and fetch share); in the Ollama
library, a name/tag convention. Anything else is left `unresolved`, never guessed, and no
automatic rule ever reaches further than `metadata_ok` -- `approved` needs a human `Approval`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .quantization import SIZE_TOKEN_RE, match_base_prefix, stem_basename
from .relation import check_relation

if TYPE_CHECKING:
    from .contracts import Approval, BaseModelSpec, Package


def decide_provenance(
    package: "Package",
    base_model: "BaseModelSpec",
    hf_tags: object,
    approvals: list["Approval"],
    *,
    hf_card_data: object = None,
    hf_targets: Sequence[str] = (),
) -> tuple[str, str | None]:
    """Return `(provenance, unresolved_reason)` for `package` against `base_model`.

    Order of decision, matching `CONTRACTS.md`:
    1. A non-GGUF package (`tensor`/`unknown` format) is always `("unresolved", "format")`.
    2. An approval whose `content` equals the package's *current* revision or manifest digest
       makes it `("approved", None)`. An approval bound to older content is void: it never
       falls back to `metadata_ok` by itself, the metadata rules below decide instead.
    3. Otherwise the metadata rules for the package's source decide.

    `hf_card_data` and `hf_targets` belong to the Hugging Face path only: the repository's
    `cardData` (the second place a relation can be declared) and the base model's whole
    package target set (`config.package_targets`), the same set the collision validator and
    the fetch use. An Ollama package ignores both.
    """
    if package.format != "gguf":
        return "unresolved", "format"

    current_content = package.revision if package.source == "huggingface" else package.manifest_digest
    for approval in approvals:
        if approval.content == current_content:
            return "approved", None

    if package.source == "huggingface":
        return _decide_huggingface(package, base_model, hf_tags, hf_card_data, hf_targets)
    return _decide_ollama(package, base_model)


def _decide_huggingface(
    package: "Package",
    base_model: "BaseModelSpec",
    hf_tags: object,
    card_data: object,
    targets: Sequence[str],
) -> tuple[str, str | None]:
    """Decide provenance for a Hugging Face package: target set, relation, file stem.

    The repository must be one of `targets`, the base model's package target set -- an
    owner-bound `repos` entry counts exactly like a generated `<base>-GGUF` name, and a
    repository outside the set is `("unresolved", "repo_name")` whatever it declares.

    The relation is `relation.check_relation`'s verdict, which search and fetch share: exactly
    one declared base, equal to this base model, with relation `quantized`. Its status is the
    `unresolved_reason` as it stands (`base_model_tag`, `relation_unknown`, `derivative`,
    `metadata_conflict`), so a reader sees the same word in a search hit and in a package.
    `hf_tags` and `card_data` are the registry's raw values of whatever shape, handed on
    untouched: a caller that cleaned them up first would turn a shape the Hub never publishes
    into a pass, which is exactly what `metadata_conflict` exists to prevent.

    The file-stem check (`quantization.match_base_prefix`) matches each weight file's own
    *basename*, never the packager's folder layout -- unsloth ships per-quant subfolders
    (`BF16/GLM-5.2-BF16-00001-of-00033.gguf`), so the folder segment must never defeat the
    prefix match, nor let an unrelated file (e.g. a draft/MTP model dropped in the same repo)
    pass under a foreign folder name. It accepts both the common `<base>-<QUANT>` convention
    and mradermacher's `<base>.<QUANT>` one.
    """
    base_name = base_model.hf_repo.split("/", 1)[1]

    if package.repo not in targets:
        return "unresolved", "repo_name"

    status, _reason = check_relation(hf_tags, card_data, base_model.hf_repo)
    if status != "quantized":
        return "unresolved", status

    weight_files = [f for f in package.files if f.role in ("weights", "weights_shard")]
    if not weight_files:
        return "unresolved", "no_weights"

    for weight_file in weight_files:
        basename = stem_basename(weight_file.name)
        if match_base_prefix(basename, base_name) is None:
            return "unresolved", "file_stem"

    return "metadata_ok", None


# A declared size token is accepted when it is within this fraction *of the declared value*
# (abs(measured - declared) <= 0.15 * declared), not of the measured parameters_b: `safetensors.
# total` counts embeddings, so a 9.653B model is legitimately tagged `9b` by its packager -- an
# exact-equality rule can never match real data (see CONTRACTS.md, "Ollama size-token
# tolerance"). 15 % is a heuristic chosen so that real registry tags pass (9.653 B tagged `9b`)
# while a genuinely wrong tag (`70b` for a 7B model) still fails -- not a measured optimum.
_SIZE_TOKEN_TOLERANCE = 0.15


def _size_token_matches(first_token: str, parameters_b: float) -> bool:
    """Whether `first_token` (the tag's first `-`-separated token) plausibly names `parameters_b`.

    `first_token` must fully match `SIZE_TOKEN_RE` (`<number>b` or `<number>m`, case-insensitive
    -- `7banana` is not a size token at all and never matches). A `b` token is compared directly
    in billions; an `m` token is converted to billions first. The declared size must be within
    15 % *of the declared value* (`abs(parameters_b - declared) <= 0.15 * declared`) of
    `parameters_b`, never `None` here -- callers gate on `parameters_b is None` first (F11).
    """
    match = SIZE_TOKEN_RE.fullmatch(first_token)
    if match is None:
        return False
    declared = float(match.group("number"))
    if match.group("unit").lower() == "m":
        declared /= 1000
    return abs(parameters_b - declared) <= _SIZE_TOKEN_TOLERANCE * declared


def _decide_ollama(package: "Package", base_model: "BaseModelSpec") -> tuple[str, str | None]:
    """Decide provenance for an Ollama package against the tag conventions in CONTRACTS.md.

    Order: the base name (`<ollama_base>:`) must prefix `ollama_name`; the base model must
    declare an `ollama_tag`; `parameters_b` must have been measured at all (F11 --
    `("unresolved", "parameters_unknown")` when it is `None`, since the size-token check below
    cannot run without a real number to compare against); the tag's first `-`-separated token
    must be a size token (`SIZE_TOKEN_RE`) whose declared size is within 15 % of `parameters_b`
    (`_size_token_matches` -- `nova:7banana` is rejected because it is not a size token at all,
    `nova:70b-q4_K_M` is rejected for a 7B model because 70 is nowhere near 7 within tolerance);
    and the full tag must equal `base_model.ollama_tag` or start with `base_model.ollama_tag +
    "-"` (token boundary, so `7b-q4_K_M` and `7b-instruct-q4_K_M` match a `7b` base tag but a
    divergent base tag such as `7b-preview` does not).
    """
    ollama_name = package.ollama_name or ""

    if not base_model.ollama_base:
        return "unresolved", "ollama_base"
    if not ollama_name.startswith(f"{base_model.ollama_base}:"):
        return "unresolved", "ollama_name"

    tag = ollama_name.split(":", 1)[1] if ":" in ollama_name else ""
    if not base_model.ollama_tag:
        return "unresolved", "ollama_tag"

    if base_model.parameters_b is None:
        return "unresolved", "parameters_unknown"

    first_token = tag.split("-", 1)[0]
    if not _size_token_matches(first_token, base_model.parameters_b):
        return "unresolved", "size_token"

    if tag != base_model.ollama_tag and not tag.startswith(f"{base_model.ollama_tag}-"):
        return "unresolved", "ollama_tag"

    return "metadata_ok", None
