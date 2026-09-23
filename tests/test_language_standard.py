"""US English is this repository's language standard; this guard is its executable form.

The rule and the word list live in `AGENTS.md`, section "Language standard". Scope here: every
module under `modelroom/`, plus the prose files `README.md`, `CONTRACTS.md`, `CHANGELOG.md`,
`AGENTS.md` and everything under `docs/`.

Four classes of finding:

1. **British spellings**, from a fixed list. A suffix rule would be wrong: `otherwise`,
   `promise`, `premise`, `exercised` and `disguise` are ordinary English, `programmer` contains
   `programme`, `greyhound` contains `grey`, and `analyses` is the correct plural of `analysis`.
   So every form is named, with the US form to write instead, and every form is anchored on
   both sides.
2. **A model's age stated as a comparison.** `older` and `newer` are correct English about a
   timestamp, a schema version or a required tool version ("Python 3.11 or newer"), and wrong
   about a model or a package: there the status is `latest`, `legacy` or `unknown`, and none of
   those needs a second thing to compare against. The rule therefore matches the phrase, not
   the line: `older`/`newer` next to a model, package or repository. `outdated` is wrong
   everywhere.
3. **A verdict on a packager.** Only the class is stated -- `publisher`, `listed packager`,
   `other` -- never `untrusted` or `trustworthy`. The two words are banned outright, including
   about input, so that no reader has to decide what was meant.
4. **German**, because every message and docstring here is read by users of the published
   package: the three umlauts, the sharp s, and a list of words that exist in no English
   sentence. `die` is the one word on the list that is also English ("the process may die"), so
   it counts only when a second German word stands on the same line.

Two deliberate limits, both the price of a guard that nobody has to argue with:

- Text inside backticks and inside URLs is skipped. Those are literals -- field names, values,
  commands, registry identifiers -- fixed by a contract or by someone else's registry. The
  price: a sentence that hides inside backticks is not read. That is also how "outside names"
  is implemented for the German words.
- Every rule reads one line at a time, so a phrase split across a line break is missed. The
  same is true of `tests/test_public_hygiene.py`, and the alternative -- rejoining paragraphs --
  loses the line number that makes a finding actionable.
"""

import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BRITISH = {
    # British form (matched case-insensitively, whole word) -> the US form to write instead.
    r"colour\w*": "color",
    r"behaviour\w*": "behavior",
    r"favour\w*": "favor",
    r"honour\w*": "honor",
    r"labour\w*": "labor",
    r"flavour\w*": "flavor",
    r"neighbour\w*": "neighbor",
    r"(?:initiali|optimi|normali|canonicali|seriali|organi|recogni|authori|summari"
    r"|prioriti|customi|minimi|maximi|utili|standardi|emphasi|apologi)s"
    r"(?:e|es|ed|ing|ation|ations|er|ers)": "-ize, -ization",
    r"analys(?:e|ed|ing)": "analyze",  # not `analyses`: that is the plural of `analysis`
    r"centre\w*": "center",
    r"catalogue\w*": "catalog",
    r"licence\w*": "license",
    r"defence\w*": "defense",
    r"offence\w*": "offense",
    r"programmes?": "program",  # not `programme\w*`: `programmer` is US English
    r"grey(?:s|ed|ing|ish|scale)?": "gray",  # not `grey\w*`: a greyhound is a dog
    r"modell(?:ed|ing)": "modeled, modeling",
    r"labell(?:ed|ing)": "labeled, labeling",
    r"cancell(?:ed|ing)": "canceled, canceling",
    r"travell(?:ed|ing)": "traveled, traveling",
    r"signall(?:ed|ing)": "signaled, signaling",
    r"fulfil(?:s|ment)?": "fulfill",
    r"whilst": "while",
    r"amongst": "among",
}

# Words that appear in no English sentence. `die` is handled separately, below.
GERMAN_WORDS = (
    "und der das nicht oder aber auch noch sind wird werden kein keine eine einen einem ein "
    "ist von beim kann muss wurde wurden gefunden Fehler Datei Ordner Rechner Pfad Bereich "
    "Sitzung Abnahme Zeile Wert Anzahl Eingabe Ausgabe Speicher"
).split()
GERMAN = r"(?i)[äöüß]|\b(?:" + "|".join(GERMAN_WORDS) + r")\b"

# What a model's age is said about. The comparison has to stand next to one of these words,
# not merely somewhere on the same line: "Python 3.11 or newer" is correct English.
AGE_SUBJECT = r"(?:models?|packages?|repos?|repositor(?:y|ies)|builds?|quantizations?)"
AGE_COMPARISON = (
    r"(?i)"
    # "an older model", "a newer GGUF package"
    r"\b(?:older|newer)\b(?=\s+(?:\w+\s+){0,2}" + AGE_SUBJECT + r"\b)"
    # "the model is older than ...", "these packages were newer"
    r"|\b" + AGE_SUBJECT + r"\s+(?:\w+\s+){0,2}(?:is|are|was|were)\s+"
    r"(?:much\s+|slightly\s+)?(?:older|newer)\b"
)

RULES = (
    # (rule name, pattern, pattern the same line must also match, what to write instead)
    (
        "British spelling",
        re.compile(r"(?i)\b(?:" + "|".join(BRITISH) + r")\b"),
        None,
        "write the US form (see BRITISH in this file)",
    ),
    (
        "model age as a comparison",
        re.compile(AGE_COMPARISON),
        None,
        "say `latest`, `legacy` or `unknown` instead of comparing two models",
    ),
    (
        "outdated",
        re.compile(r"(?i)\b(?:outdated|out-of-date)\b"),
        None,
        "say `legacy` with its `successor`, or `unknown`",
    ),
    (
        "packager valuation",
        re.compile(r"(?i)\b(?:untrusted|untrustworthy|trustworthy)\b"),
        None,
        "state the owner class (`publisher`, `listed packager`, `other`), never a verdict",
    ),
    ("German", re.compile(GERMAN), None, "write it in English"),
    (
        "German, with a word that is also English",
        re.compile(r"(?i)\bdie\b"),
        re.compile(GERMAN),
        "write it in English (`die` counts only next to another German word)",
    ),
)

# Exception list: findings that stay for now, one entry per file, rule and word, each with the
# number of occurrences it covers and the reason. Keyed by the word and not by a line number on
# purpose -- a line number moves with every edit above it, and these modules are being changed
# elsewhere at the same time. The count is what keeps the entry from covering a second, new
# violation of the same kind: occurrence number two is reported like any other finding, and
# `test_every_exception_still_exempts_something` turns red once the count is too high.
PENDING: dict[tuple[str, str, str], tuple[int, str]] = {
    # Empty since the search work package converted `provenance.py` and `ollama.py` and the
    # merge converted `guided_contracts.py`. An entry looks like
    # ("modelroom/<file>.py", "<rule name>", "<word>"): (<count>, "<reason>").
}

CODE_SPAN = re.compile(r"``[^`]*``|`[^`]*`")
URL = re.compile(r"""https?://[^\s"'`)\]>,]+""")


def scanned_files() -> list[Path]:
    """Every module of the package, plus the prose it ships and documents itself with."""
    files = sorted((REPO / "modelroom").rglob("*.py"))
    files += [REPO / name for name in ("README.md", "CONTRACTS.md", "CHANGELOG.md", "AGENTS.md")]
    files += sorted((REPO / "docs").rglob("*.md"))
    return [path for path in files if path.exists()]


def prose_of(line: str) -> str:
    """The part of a line this standard governs: no code spans, no URLs."""
    return URL.sub(" ", CODE_SPAN.sub(" ", line))


def matches_in(line: str, rule_name: str) -> list[str]:
    """Every word the named rule finds in one already-stripped line."""
    for name, pattern, context, _hint in RULES:
        if name != rule_name:
            continue
        if context is not None and not context.search(line):
            return []
        return [match.group(0) for match in pattern.finditer(line)]
    raise AssertionError(f"no rule named {rule_name}")


def raw_findings() -> list[tuple[str, int, str, str]]:
    """(path, line number, rule, word) for every match in scope, exceptions included."""
    found = []
    for path in scanned_files():
        rel = path.relative_to(REPO).as_posix()
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = prose_of(raw)
            for name, _pattern, _context, _hint in RULES:
                found += [(rel, line_no, name, word) for word in matches_in(line, name)]
    return found


def findings() -> list[str]:
    hints = {name: hint for name, _pattern, _context, hint in RULES}
    covered: Counter[tuple[str, str, str]] = Counter()
    hits = []
    for rel, line_no, name, word in raw_findings():
        key = (rel, name, word.lower())
        allowance = PENDING.get(key, (0, ""))[0]
        covered[key] += 1
        if covered[key] <= allowance:
            continue
        hits.append(f"{rel}:{line_no}: {name}: {word!r} -- {hints[name]}")
    return hits


def over_stated_exceptions() -> list[str]:
    """Entries that exempt more occurrences than the repository still has."""
    counts = Counter((rel, name, word.lower()) for rel, _no, name, word in raw_findings())
    problems = []
    for key, (expected, _reason) in PENDING.items():
        actual = counts.get(key, 0)
        if actual < expected:
            rel, name, word = key
            problems.append(
                f"{rel}: {name}: {word!r}: exempts {expected}, found {actual} "
                "-- lower the count or delete the entry"
            )
    return problems


def test_package_and_prose_are_us_english():
    hits = findings()
    assert hits == [], "language standard (AGENTS.md) not met:\n" + "\n".join(hits)


def test_every_exception_still_exempts_something():
    """An exception that exempts nothing is a false statement about the repository."""
    problems = over_stated_exceptions()
    assert problems == [], "exception list out of date:\n" + "\n".join(problems)


def test_the_guard_reports_a_planted_violation():
    """A guard that has never gone red says nothing when it is green."""
    planted = {
        "British spelling": [
            "the behaviour of the catalogue is initialised",
            "the colour was normalised and the licence renewed",
        ],
        "model age as a comparison": [
            "an older model is listed",
            "a newer package replaces it",
            "the model is older than its successor",
        ],
        "outdated": ["the package is outdated"],
        "packager valuation": ["the packager is untrusted", "nothing here is trustworthy"],
        "German": ["Fehler beim Laden", "die Datei wurde nicht gefunden", "ein Rechner"],
        "German, with a word that is also English": ["die Datei wurde nicht gefunden"],
    }
    for rule_name, lines in planted.items():
        for line in lines:
            assert matches_in(prose_of(line), rule_name), (rule_name, line)


def test_ordinary_english_is_not_a_finding():
    """The lists are fixed lists because suffix rules catch correct English."""
    clean = [
        "otherwise the promise is exercised on the premise of a disguise",
        "a programmer writes the analysis; these analyses agree on the emphasis",
        "a greyhound is not a spelling mistake",
        "this package requires Python 3.11 or newer",
        "llmfit is older than the minimum version this release needs",
        "the run is not newer than the stored snapshot's run_at",
        "the sha is only ever trusted when it matches the pinned revision",
        "the process may die before the lock is released",
    ]
    for line in clean:
        for name, _pattern, _context, _hint in RULES:
            assert matches_in(prose_of(line), name) == [], (name, line)


def test_literals_and_urls_are_not_read_as_prose():
    """A field name, a value or a registry identifier is contract, not prose."""
    assert prose_of("the state is `not comparable`").split() == ["the", "state", "is"]
    assert matches_in(prose_of("see `catalogue_of(x)`"), "British spelling") == []
    assert matches_in(prose_of("see ``catalogue_of(x)``"), "British spelling") == []
    assert matches_in(prose_of("https://example.invalid/das/und/die"), "German") == []
    # A URL ends where the line's own syntax ends; it does not swallow what follows.
    after_url = prose_of("""url="https://example.invalid"; message = "colour" """)
    assert matches_in(after_url, "British spelling") == ["colour"]


if __name__ == "__main__":
    problems = findings() + over_stated_exceptions()
    for problem in problems:
        print(problem)
    raise SystemExit(1 if problems else 0)
