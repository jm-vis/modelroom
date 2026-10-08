"""The card follows the catalog: the dataset card, the banner and the social image carry the numbers of `catalog.json` (AP9-KA-C).

    python scripts/catalog_card.py --check
    python scripts/catalog_card.py [--write] [--render] [--upload [--dataset <id> | --probe]]

Models (rows of `models`), accounts (rows of `publishers`) and the size (`round(bytes / 1000)` KB) of the catalog stand in the
card and in the first two pills of both pictures. `--check` reports each place that disagrees (`ALARM card: ...`, exit 1);
`--write` brings the files to the numbers (a pill widens by `round(font * 0.57)` per character, the pills after it move);
`--render` makes the PNGs with a headless Chrome (`MODELROOM_CHROME`); `--upload` sends the card (as `README.md`) and the banner
in one process. `--write --render --upload` run in that order, the first failure stops. Exit 1: a difference or a failed
upload; exit 2: the call, an input or the browser is wrong. Allowed calls: one mode alone, or `--write --render --upload` together.
The singular is not in the patterns: `1 models` is written, a file that says `1 model` is refused (`CardError`).
A maintainer tool, not in the wheel; it never commits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the other modules of the scripts sit beside this one
import catalog_publish as publish  # noqa: E402
import catalog_publish_rules as P  # noqa: E402
from catalog_scan import DiskFiles  # noqa: E402

CARD, BANNER, SOCIAL = "docs/dataset-card.md", "docs/assets/catalog-banner.svg", "docs/assets/catalog-social.svg"
DATASET, CHROME_ENV = "jm-vis/modelroom-catalog", "MODELROOM_CHROME"
CHROME_DEFAULT = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
_BOLD = re.compile(r"\*\*(\d+) models · (\d+) publisher accounts watched · one JSON file \((\d+) KB\)")
_SENTENCE = re.compile(r"reads the (\d+) publisher accounts")
_GROUP = re.compile(r'(<g class="chip" font-size=")(\d+)(">)(.*?)(</g>)', re.S)
_RECT = re.compile(r'(<rect x=")(\d+)(" y="\d+" width=")(\d+)(")')
_TEXT = re.compile(r'(<text x=")([\d.]+)(" y="[\d.]+" fill="#FFFFFF" text-anchor="middle">)([^<]*)(</text>)')
_MODELS, _PUBLISHERS = re.compile(r"(\d+) models"), re.compile(r"(\d+) publishers watched")
_SIZE = re.compile(r'<svg [^>]*width="(\d+)" height="(\d+)"')


class CardError(Exception):
    """The call, an input or the browser is wrong: exit 2, the remaining steps are not run."""


class UploadFailed(Exception):
    """The upload could not be made: exit 1, nothing more."""


class CardNumbers(NamedTuple):
    models: int
    publishers: int
    kilobytes: int


def numbers(catalog: bytes) -> CardNumbers:
    try:
        document = json.loads(catalog.decode("utf-8"))
        models, publishers = document["models"], document["publishers"]
        if not isinstance(models, list) or not isinstance(publishers, list):
            raise TypeError
    except (ValueError, KeyError, TypeError) as exc:
        raise CardError("the catalog is not a catalog: it has no valid models and publishers") from exc
    return CardNumbers(len(models), len(publishers), round(len(catalog) / 1000))


def _once(pattern: re.Pattern[str], text: str, what: str) -> re.Match[str]:
    found = list(pattern.finditer(text))
    if len(found) != 1:
        raise CardError(f"{what}: the pattern hits {len(found)} times, not once; the file changed its shape")
    return found[0]


def card_text(text: str, n: CardNumbers) -> str:
    bold, sentence = _once(_BOLD, text, "card line"), _once(_SENTENCE, text, "card sentence")
    if bold.end() > sentence.start():
        raise CardError("card: the sentence stands before the bold line; the file changed its shape")
    line = f"**{n.models} models · {n.publishers} publisher accounts watched · one JSON file ({n.kilobytes} KB)"
    return text[:bold.start()] + line + text[bold.end():sentence.start()] + f"reads the {n.publishers} publisher accounts" + text[sentence.end():]


def _pills(text: str, what: str):
    """The chip group of a picture: font size, the three rects and the three texts (the first two are the numbers)."""
    group = _once(_GROUP, text, what + " chips")
    body = group.group(4)
    rects, texts = list(_RECT.finditer(body)), list(_TEXT.finditer(body))
    if (len(rects) != 3 or len(texts) != 3 or not _MODELS.fullmatch(texts[0].group(4)) or not _PUBLISHERS.fullmatch(texts[1].group(4))
            or _MODELS.fullmatch(texts[2].group(4)) or _PUBLISHERS.fullmatch(texts[2].group(4))):  # exactly one pill of each number
        raise CardError(f"{what}: the three pills are not as expected; the file changed its shape")
    return group, int(group.group(2)), rects, texts


def _fmt(value: float, one_decimal: bool) -> str:
    return f"{value:.1f}" if one_decimal else str(int(value)) if value == int(value) else repr(value)


def svg_text(text: str, n: CardNumbers, what: str = "picture") -> str:
    group, font, rects, texts = _pills(text, what)
    step = round(font * 0.57)
    new = (f"{n.models} models", f"{n.publishers} publishers watched")
    d1, d2 = (step * (len(new[i]) - len(texts[i].group(4))) for i in (0, 1))
    one_decimal = any(m.group(2).endswith(".0") for m in texts)  # the banner writes 398.0, the social image 591
    rect_edit = [(0, d1), (d1, d2), (d1 + d2, 0)]  # (move of x, change of width) per pill
    text_move = [d1 / 2, d1 + d2 / 2, d1 + d2]
    edits = [(m.span(), f"{m.group(1)}{int(m.group(2)) + move}{m.group(3)}{int(m.group(4)) + grow}{m.group(5)}") for m, (move, grow) in zip(rects, rect_edit)]
    edits += [(m.span(), f"{m.group(1)}{_fmt(float(m.group(2)) + text_move[i], one_decimal)}{m.group(3)}{(*new, m.group(4))[i]}{m.group(5)}")
              for i, m in enumerate(texts)]
    body = group.group(4)
    for (a, b), replacement in sorted(edits, reverse=True):
        body = body[:a] + replacement + body[b:]
    return text[:group.start(4)] + body + text[group.end(4):]


def _found(card: str, banner: str, social: str) -> list[tuple[str, int, int]]:
    """(place, number in the file, index into CardNumbers) for every place the numbers stand."""
    bold, sentence = _once(_BOLD, card, "card line"), _once(_SENTENCE, card, "card sentence")
    places = [("dataset-card.md models", int(bold.group(1)), 0), ("dataset-card.md accounts", int(bold.group(2)), 1),
              ("dataset-card.md size in KB", int(bold.group(3)), 2), ("dataset-card.md sentence accounts", int(sentence.group(1)), 1)]
    for name, text in (("catalog-banner.svg", banner), ("catalog-social.svg", social)):
        texts = _pills(text, name)[3]
        places += [(f"{name} models", int(_MODELS.fullmatch(texts[0].group(4)).group(1)), 0),
                   (f"{name} accounts", int(_PUBLISHERS.fullmatch(texts[1].group(4)).group(1)), 1)]
    return places


def _read(files, root: Path, name: str) -> bytes:
    try:
        return files.read(root / name)
    except OSError as exc:
        raise CardError(f"{name}: cannot be read") from exc


def _text(data: bytes, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CardError(f"{Path(name).name}: not valid UTF-8") from exc


def _tree(files, root: Path):
    catalog = numbers(_read(files, root, "modelroom/catalog.json"))
    return catalog, [_text(_read(files, root, name), name) for name in (CARD, BANNER, SOCIAL)]


def check(files, root: Path) -> list[str]:
    n, texts = _tree(files, root)
    return [f"ALARM card: {place} says {said}, the catalog has {n[index]}" for place, said, index in _found(*texts) if said != n[index]]


def write(files, root: Path, out) -> None:
    n, texts = _tree(files, root)
    results = [card_text(texts[0], n), svg_text(texts[1], n, "banner"), svg_text(texts[2], n, "social image")]  # all before any file
    for name, old, new in zip((CARD, BANNER, SOCIAL), texts, results):
        print(f"{'written' if new != old else 'unchanged'} {name}", file=out)
        new != old and files.write(root / name, new.encode("utf-8"))


def render(svg: Path, png: Path, run, env, files=DiskFiles(), isfile=os.path.isfile) -> None:
    chrome = env.get(CHROME_ENV) or CHROME_DEFAULT
    if not isfile(chrome):
        raise CardError(f"no browser at {chrome}: set {CHROME_ENV} to a Chrome; the card is checked but not rendered without one")
    head = _SIZE.search(_text(files.read(svg), svg.name)[:600])
    if not head:
        raise CardError(f"{svg.name}: no width and height in the svg head")
    child = {k: v for k, v in env.items() if k not in publish.TOKEN_VARIABLES}
    with tempfile.TemporaryDirectory(prefix="catalog-render-") as work, tempfile.TemporaryDirectory(prefix="catalog-profile-") as profile:
        shot = Path(work) / png.name  # a fresh profile each call: a second run with the same one writes nothing
        args = [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={head.group(1)},{head.group(2)}",
                f"--screenshot={shot}", f"--user-data-dir={profile}", svg.resolve().as_uri()]
        code = run(args, child)  # an OSError (no such program) ends the call in main
        if code != 0 or not shot.is_file() or shot.stat().st_size <= 10_000:
            raise CardError(f"{png.name}: the browser made no picture of more than 10 KB (exit {code})")
        files.write(png, shot.read_bytes())


def _words(root: Path, env) -> list[str]:
    location = env.get(publish.PRIVATE_WORDS_ENV, "").strip()
    if not location:
        raise UploadFailed(f"{publish.PRIVATE_WORDS_ENV} is not set: it names the file of private words")
    try:
        path = Path(location)
        if path.resolve().is_relative_to(root.resolve()):
            raise UploadFailed(f"the file {publish.PRIVATE_WORDS_ENV} names lies inside the repository; keep it outside")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise UploadFailed(f"the file {publish.PRIVATE_WORDS_ENV} names cannot be read") from exc
    return sorted({ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}, key=len, reverse=True)


def _hygiene(root: Path, data: bytes, env) -> None:
    words = _words(root, env)
    try:
        spec = importlib.util.spec_from_file_location("release_check", root / "scripts" / "release-check.py")
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        findings = gate.scan_data("README.md", "README.md", data, gate.with_private_words(gate.load_negative_list(), words))
    except (OSError, AttributeError, ImportError, SyntaxError, TypeError) as exc:
        raise UploadFailed(f"the hygiene check cannot run: {type(exc).__name__}") from exc
    if findings:
        raise UploadFailed("the card fails the hygiene check, nothing is uploaded: " + "; ".join(findings[:5]))


def upload(root: Path, run, env, files=DiskFiles(), dataset: str = DATASET) -> None:
    try:
        readme, banner = files.read(root / CARD), files.read(root / "docs/assets/catalog-banner.png")
    except OSError as exc:
        raise UploadFailed("the card or the banner picture cannot be read") from exc
    _hygiene(root, readme, env)
    token = env.get(publish.TOKEN_PATH_ENV, "")
    if not token or not os.path.isfile(token) or not os.access(token, os.R_OK):
        raise UploadFailed(f"{publish.TOKEN_PATH_ENV} is not set or names no readable file")
    child = {k: v for k, v in env.items() if k not in publish.TOKEN_VARIABLES}
    child[publish.TOKEN_PATH_ENV] = token
    try:
        with tempfile.TemporaryDirectory(prefix="catalog-card-") as folder:
            (Path(folder) / "README.md").write_bytes(readme)
            (Path(folder) / "catalog-banner.png").write_bytes(banner)
            again = files.read(Path(folder) / "README.md")
            if again != readme:
                raise UploadFailed("the upload file does not hold the bytes it was made from")
            _hygiene(root, again, env)
            code = run(P.upload_folder_args(dataset, folder, "card follows the catalog"), child)
    except OSError as exc:
        raise UploadFailed(f"the upload folder could not be made or the process not started: {type(exc).__name__}") from exc
    if code != 0:
        raise UploadFailed(f"the upload failed (exit {code})")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catalog_card.py", description="the card follows the catalog")
    for flag in ("check", "write", "render", "upload", "probe"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--dataset")
    return parser


def main(argv: list[str] | None = None, *, root: Path | None = None, files=None, run=None, env=None, out=None, err=None) -> int:
    root, out, err = (root or Path(__file__).resolve().parent.parent).resolve(), out or sys.stdout, err or sys.stderr
    try:
        a = _parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    allowed = {(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 1, 1, 1)}  # (check, write, render, upload): one alone, or all three steps
    if (int(a.check), int(a.write), int(a.render), int(a.upload)) not in allowed or ((a.dataset or a.probe) and not a.upload) or (a.dataset and a.probe):
        print("error: give --check, --write, --render or --upload alone, or --write --render --upload together (--dataset or --probe only with --upload)", file=err)
        return 2
    files, run, env = files or DiskFiles(), run or publish.default_run, os.environ if env is None else env
    dataset = publish.PROBE_DATASET if a.probe else a.dataset or DATASET
    try:
        if a.check:
            lines = check(files, root)
            print("\n".join(lines), file=out) if lines else None
            return 1 if lines else 0
        if a.write:
            write(files, root, out)
        if a.render:
            for svg in (BANNER, SOCIAL):
                render(root / svg, (root / svg).with_suffix(".png"), run, env, files)
                print(f"rendered {svg[:-4]}.png", file=out)
        if a.upload:
            upload(root, run, env, files, dataset)
            print(f"uploaded the card and the banner to {dataset}", file=out)
    except CardError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except UploadFailed as exc:
        print(f"failed: {exc}", file=err)
        return 1
    except OSError as exc:  # the browser or the upload program cannot be started
        print(f"error: a process cannot be started or a file written ({type(exc).__name__})", file=err)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
