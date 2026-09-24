"""The answer file: `modelroom --answers <file>` takes the dialog's answers from TOML.

One key per question, as CONTRACTS.md, "Guided mode", lists them; `schema_version = 1` on top.
A question the file has no answer for ends the run with exit `2` and names the question, so a
self-test or a CI run never silently answers something it did not mean to
(`dialog.FileAsker`). Nothing else about the guided mode changes: the same steps run in the
same order and print the same lines.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .contracts import SchemaVersionError, check_schema_version

ANSWERS_SCHEMA_VERSION = 1
# Half-open, same convention as every other reader in this package.
ANSWERS_SCHEMA_RANGE: tuple[int, int] = (1, 2)


class AnswerFileError(Exception):
    """The answer file does not read, or one of its answers is not a shape a question can take."""


def _is_answer(value: object) -> bool:
    """The four shapes a question's answer can have: text, a choice, a switch, a list of choices."""
    if isinstance(value, bool) or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    return isinstance(value, list) and all(isinstance(entry, str) for entry in value)


def read_answers(path: Path) -> dict[str, object]:
    """Every answer in the file at `path`, by question key; `schema_version` is not an answer.

    Raises `AnswerFileError` when the file cannot be read or holds an answer of another shape,
    and `SchemaVersionError` when its `schema_version` is missing or outside the accepted range.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise AnswerFileError(f"{path}: cannot read the answer file: {exc}") from exc
    version = raw.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"{path}: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, ANSWERS_SCHEMA_RANGE, str(path))
    answers = {key: value for key, value in raw.items() if key != "schema_version"}
    for key, value in answers.items():
        if not _is_answer(value):
            raise AnswerFileError(
                f"{path}: the answer {key!r} is not text, a whole number, true/false or a list of texts"
            )
    return answers
