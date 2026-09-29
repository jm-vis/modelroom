"""The one sink for names: what of a machine's name and of foreign text may reach a file (AP9-FR, FR-N).

Every place modelroom writes or exports a file hands its object through the matching function here
first; there is no second way (CONTRACTS.md, "Neutral names"). The rules, complete -- everything else
is dropped, never passed on:

| field | allowed | otherwise |
|---|---|---|
| machine key `[machines.<k>]` | `NEUTRAL_RE`, or a host alias at the row of the profile it belongs to | `this_machine` for this computer's row, `neutral_name(profile)`, else `computer-<8 random hex>` |
| `display_name` | `NEUTRAL_RE`, or the host alias of that profile | `neutral_name(profile)` |
| `MeasurementRecord.ollama_name` | the name built from the known package (the load test's, else the one of the folder's snapshot the record's digest names); without one only `hf.co/<hf_repo>:<tag>` of the record's repository | `None` |
| `comparable_reason` | a sentence of `COMPARABLE_REASONS`, or the context sentence with two whole numbers | the sentence of the local text's beginning, else `UNKNOWN_COMPARABLE` |
| `validity_reason` | the sentences `measurements.measurement_invalid_reason` forms | `UNKNOWN_VALIDITY` |
| `daemon_version` | exactly `X.Y.Z` | `None` |
| `runs[].done_reason` | a reason Ollama gives (`DONE_REASONS`) | `unknown` |

The functions are pure. `modelroom/neutral_sweep.py` runs the same rules over the files a folder
already holds, and is the only part with I/O.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from .measurements import LEGACY_REASON, ExportObject, MeasurementRecord
from .profile import HardwareProfile

if TYPE_CHECKING:  # pragma: no cover - only the type of an argument
    from .contracts import Package

NEUTRAL_RE = re.compile(r"^(laptop|workstation|server|computer)-[0-9a-f]{8}$")
THIS_MACHINE_RE = re.compile(r"^computer-[0-9a-f]{8}$")
# A host alias (FR-B1): letters, digits and dashes, no dot (no address, no domain), no colon, no `@`.
ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_CONTEXT_RE = re.compile(r"^the daemon loaded context \d+, not the ranking's \d+$")
_HF_NAME_RE = re.compile(r"^hf\.co/(?P<repo>[^:\s]+):[A-Za-z0-9_.-]+$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SHORT = 8
_ATTEMPTS = 8

FAILED = "the files of this folder could not be cleaned of machine names; nothing else was written"
UNKNOWN_COMPARABLE = "the measurement is not comparable (reason not kept)"
UNKNOWN_VALIDITY = "the measurement is not valid (reason not kept)"
ANOTHER_MODEL = "another model was loaded during the measurement"
NO_MODEL_SEEN = "no observation of the measured model"
NO_OBSERVATION = "no observation of /api/ps was made"
DIGEST_CHANGED = "the digest of the measured model changed during the measurement"
COMPARABLE_REASONS = (ANOTHER_MODEL, NO_MODEL_SEEN, NO_OBSERVATION, DIGEST_CHANGED, LEGACY_REASON)
# The beginning of a local text of `loadtest.not_comparable_reason`, and the sentence it becomes.
_REASON_STARTS = (
    ("/api/ps showed ", ANOTHER_MODEL),
    ("no observation of the measured model", NO_MODEL_SEEN),
    ("the digest of ", DIGEST_CHANGED),
)
DONE_REASONS = ("stop", "length", "load", "unload", "unknown")
_DONE = "(?:" + "|".join(DONE_REASONS) + ")"
_VALIDITY_RES = tuple(
    re.compile(pattern)
    for pattern in (
        r"^\d+ measured runs, protocol requires \d+$",
        rf"^run \d+: done_reason '{_DONE}', expected '{_DONE}'$",
        r"^run \d+: eval_count \d+, expected \d+$",
        r"^run \d+: eval_duration is not above zero$",
        r"^run \d+: prompt_eval_count \d+ \+ \d+ exceeds num_ctx \d+$",
    )
)


class NeutralError(Exception):
    """An object could not be brought through the sink, or a folder could not be swept: nothing is
    written (exit `1`, with `FAILED`)."""


# --- names ---------------------------------------------------------------------------------------


def is_neutral(name: str | None) -> bool:
    """`<class>-<8 hex>`: the only shape of a machine name modelroom writes on its own."""
    return name is not None and NEUTRAL_RE.fullmatch(name) is not None


def is_alias(name: str | None) -> bool:
    """A host alias a user gave: `ALIAS_RE`, not only digits (ssh would read an address), and not in
    the shape of a neutral name, so that an alias and a neutral key never collide (Codex r7-1)."""
    return name is not None and ALIAS_RE.fullmatch(name) is not None and not name.isdigit() and not is_neutral(name)


def random_key(token: Callable[[int], str] = secrets.token_hex) -> str:
    """`computer-<8 random hex>`: a key nothing about the machine can be read from."""
    return f"computer-{token(_SHORT // 2)}"


def neutral_name(profile_id: str, machine_class: str | None) -> str:
    """`<class>-<first 8 hex of the profile_id>`; the class `computer` where it is unknown."""
    kind = machine_class if machine_class in ("laptop", "workstation", "server") else "computer"
    return f"{kind}-{profile_id[:_SHORT]}"


def old_key(hostname: str | None) -> str | None:
    """The key modelroom 0.1.0 gave this computer, formed from its host name as read live; never stored."""
    slug = _SLUG_RE.sub("-", (hostname or "").lower()).strip("-")
    return slug or None


def display_name_for(profile_id: str, machine_class: str | None, current: str | None, alias: str | None) -> str:
    """The `display_name` a profile may carry: its alias, a neutral name as it is, else `neutral_name`."""
    if alias is not None and is_alias(alias):
        return alias
    return current if is_neutral(current) else neutral_name(profile_id, machine_class)


def neutral_profile(profile: HardwareProfile, alias: str | None = None) -> HardwareProfile:
    """`profile` with the `display_name` the sink allows; nothing else of it changes."""
    name = display_name_for(profile.profile_id, profile.machine_class, profile.display_name, alias)
    return profile if name == profile.display_name else profile.model_copy(update={"display_name": name})


def renamed_note(new: str) -> str:
    """The note of a machine the sweep renamed: only the new name, never the old one."""
    return f"machine renamed to {new} (no host names in files)"


# --- measurement records -------------------------------------------------------------------------


def neutral_version(value: str | None) -> str | None:
    """A daemon version as `X.Y.Z`, or `None`: any addition drops the whole value (Codex r3-3)."""
    return value if value is not None and _VERSION_RE.fullmatch(value) else None


def comparable_reason(reason: str | None) -> str | None:
    """A fixed sentence: kept, mapped from the beginning of a known local text, or the unknown one."""
    if reason is None or reason in COMPARABLE_REASONS or _CONTEXT_RE.fullmatch(reason):
        return reason
    for start, sentence in _REASON_STARTS:
        if reason.startswith(start):
            return sentence
    return UNKNOWN_COMPARABLE


def validity_reason(reason: str | None) -> str | None:
    """A sentence `measurement_invalid_reason` forms, `LEGACY_REASON`, or the unknown one."""
    if reason is None or reason == LEGACY_REASON or any(pattern.fullmatch(reason) for pattern in _VALIDITY_RES):
        return reason
    return UNKNOWN_VALIDITY


def package_name(package: "Package") -> str | None:
    """The name modelroom builds from the package itself (`pull_choice.package_pull_name`, A2)."""
    from .pull_choice import package_pull_name  # imported here: that module reaches the screen

    return package_pull_name(package)


def record_name(name: str | None, hf_repo: str | None) -> str | None:
    """Without a known package, only `hf.co/<hf_repo>:<tag>` of the record's own repository."""
    match = _HF_NAME_RE.fullmatch(name) if name is not None else None
    return name if match is not None and hf_repo is not None and match.group("repo") == hf_repo else None


def neutral_record(record: MeasurementRecord, package: "Package | None" = None) -> MeasurementRecord:
    """`record` with every field of the table in its allowed form; the measurement itself stays."""
    name = package_name(package) if package is not None else record_name(record.ollama_name, record.package.hf_repo)
    runs = [run if run.done_reason in DONE_REASONS else run.model_copy(update={"done_reason": "unknown"}) for run in record.runs]
    update = {
        "ollama_name": name,
        "daemon_version": neutral_version(record.daemon_version),
        "comparable_reason": comparable_reason(record.comparable_reason),
        "validity_reason": validity_reason(record.validity_reason),
        "runs": runs,
    }
    changed = {key: value for key, value in update.items() if getattr(record, key) != value}
    if not changed:
        return record
    try:
        return MeasurementRecord.model_validate({**record.model_dump(mode="json"), **_json(changed)})
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        raise NeutralError(FAILED) from exc


def _json(changed: dict) -> dict:
    return {key: [run.model_dump(mode="json") for run in value] if key == "runs" else value for key, value in changed.items()}


def known_package(record: MeasurementRecord, packages: "Sequence[Package]") -> "Package | None":
    """The package of the folder's own snapshot the record's reference names, by digest; else `None`.

    The identity modelroom itself fetched, never a name the daemon gave: so a record keeps the name
    of its package in every folder that knows the package, whichever command writes it.
    """
    ref = record.package
    for package in packages:
        if ref.content_source == "ollama" and package.source == "ollama" and package.manifest_digest == ref.ollama_manifest_digest:
            return package
        digests = {file.digest for file in package.files if file.digest}
        if ref.content_source == "huggingface" and package.repo == ref.hf_repo and ref.hf_file_digest in digests:
            return package
    return None


def neutral_records(records: "Sequence[MeasurementRecord]", packages: "Sequence[Package]" = ()) -> list[MeasurementRecord]:
    """Every record through the sink, each with its package where the folder knows it."""
    return [neutral_record(record, known_package(record, packages)) for record in records]


def neutral_export(export: ExportObject, alias: str | None = None, packages: "Sequence[Package]" = ()) -> ExportObject:
    """An export with its profile and every record through the sink (export, and import before planning)."""
    return ExportObject(
        schema_version=export.schema_version,
        profile=neutral_profile(export.profile, alias),
        measurements=neutral_records(export.measurements, packages),
    )


# --- machine keys ----------------------------------------------------------------------------------


@dataclass
class Names:
    """What the key rule needs to know besides the rows: this computer and the user's aliases.

    `this_machine` is the stored key (`None` before the first need), `make_this` makes and stores it;
    `bound` is the profile this results folder binds this computer to; `old` its 0.1.0 key.
    """

    this_machine: str | None = None
    make_this: Callable[[], str] = random_key
    bound: str | None = None
    aliases: Mapping[str, str] = field(default_factory=dict)
    old: str | None = None


def own_row(entries: Mapping[str, str | None], names: Names) -> str | None:
    """The row that is this computer: the one of the bound profile (Codex r5-2), else its 0.1.0 key."""
    bound = sorted(key for key, profile in entries.items() if names.bound is not None and profile == names.bound)
    if bound:
        return names.this_machine if names.this_machine in bound else bound[0]
    return names.old if names.old is not None and names.old in entries else None


def _kept(key: str, profile: str | None, aliases: Mapping[str, str]) -> bool:
    """A neutral key, or an alias at the row of the profile it belongs to (Codex r6-1)."""
    return is_neutral(key) or (profile is not None and is_alias(key) and aliases.get(profile) == key)


def free_key(taken: set[str], new_key: Callable[[], str] = random_key) -> str:
    """A random neutral key no row carries; a source that keeps repeating itself is `NeutralError`."""
    for _ in range(_ATTEMPTS):
        key = new_key()
        if key not in taken:
            return key
    raise NeutralError(FAILED)


def machine_keys(
    entries: Mapping[str, str | None], classes: Mapping[str, str], names: Names, new_key: Callable[[], str] = random_key
) -> dict[str, str]:
    """`old key -> new key` for every row (`entries`: key -> its `profile`, `classes`: profile -> class).

    This computer's row becomes `this_machine`, and `this_machine` at any other row yields to it; a
    neutral key and an alias at its own profile stay; every other row gets `neutral_name` of its
    profile, or a random neutral key where it has none or that name is taken.
    """
    own = own_row(entries, names)
    this = (names.this_machine or names.make_this()) if own is not None else None
    result = {key: key for key, profile in entries.items() if key != own and key != this and _kept(key, profile, names.aliases)}
    if own is not None:
        result[own] = this
    taken = set(result.values())
    for key, profile in sorted(entries.items()):
        if key in result:
            continue
        wanted = neutral_name(profile, classes.get(profile)) if profile is not None else None
        result[key] = wanted if wanted is not None and wanted not in taken else free_key(taken, new_key)
        taken.add(result[key])
    return result


def key_for(
    profile_id: str, machine_class: str | None, display_name: str | None, taken: set[str], alias: str | None = None
) -> str:
    """The key of a new row (import, a profile found in the folder, a machine entered by hand): the
    alias where the SSH way gives one, else the profile's neutral name, else a random neutral key."""
    if alias is not None and is_alias(alias) and alias not in taken:
        return alias
    wanted = display_name_for(profile_id, machine_class, display_name, None)
    return wanted if wanted not in taken else free_key(taken)


__all__ = [
    "ALIAS_RE",
    "COMPARABLE_REASONS",
    "FAILED",
    "NEUTRAL_RE",
    "THIS_MACHINE_RE",
    "UNKNOWN_COMPARABLE",
    "UNKNOWN_VALIDITY",
    "Names",
    "NeutralError",
    "comparable_reason",
    "display_name_for",
    "free_key",
    "is_alias",
    "is_neutral",
    "key_for",
    "known_package",
    "machine_keys",
    "neutral_export",
    "neutral_name",
    "neutral_profile",
    "neutral_record",
    "neutral_records",
    "neutral_version",
    "old_key",
    "own_row",
    "package_name",
    "random_key",
    "record_name",
    "renamed_note",
    "validity_reason",
]
