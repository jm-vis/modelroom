"""Tests for `scripts/catalog_card.py`: the dataset card, the banner and the social image follow the catalog (AP9-KA-C).

The goldens are the real files of two commits, written once into `tests/fixtures/catalog_card/old` (80 models, 54 accounts,
65 KB) and `.../new` (231, 56, 171 KB). Nothing here opens a socket or starts Chrome or an upload: the one outside process is
a recording function that keeps the argument list and the environment it was given (never a token value) and writes the
screenshot a browser would write.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the card tool is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))
card = importlib.import_module("catalog_card")
P = importlib.import_module("catalog_publish_rules")

FIX = REPO / "tests" / "fixtures" / "catalog_card"
OLD, NEW = card.CardNumbers(80, 54, 65), card.CardNumbers(231, 56, 171)
NAMES = ("dataset-card.md", "catalog-banner.svg", "catalog-social.svg")
TOKEN_TEXT = "placeholder-token-content"
PNG = b"\x89PNG" + b"\0" * 20000


def golden(version: str, name: str) -> bytes:
    return (FIX / version / name).read_bytes()


def catalog_bytes(models: int, publishers: int, size: int) -> bytes:
    base = json.dumps({"models": [{}] * models, "publishers": [{}] * publishers, "pad": ""}).encode()
    return base[:-2] + b"x" * (size - len(base)) + base[-2:]


class Runner:
    """The outside processes: records (arguments, environment); writes the screenshot; sees the upload folder as it is."""

    def __init__(self, png: bytes = PNG, codes: dict | None = None) -> None:
        self.calls, self.folders, self.png, self.codes = [], [], png, codes or {}

    def __call__(self, args, env):
        self.calls.append((list(args), dict(env)))
        if args[0] == "uvx":
            folder = Path(args[6])
            self.folders.append({p.name: p.read_bytes() for p in folder.iterdir()})
        else:
            shot = next(a for a in args if a.startswith("--screenshot="))
            Path(shot.split("=", 1)[1]).write_bytes(self.png)
        return self.codes.get(args[0], 0)


@pytest.fixture
def repo(tmp_path):
    """A tree of the card's files at the old numbers and a catalog at the new ones; the maintainer's secrets beside it."""
    root = tmp_path / "work"
    (root / "modelroom").mkdir(parents=True)
    (root / "docs" / "assets").mkdir(parents=True)
    (root / "modelroom" / "catalog.json").write_bytes(catalog_bytes(231, 56, 171215))
    shutil.copy(FIX / "old" / NAMES[0], root / "docs" / NAMES[0])
    for name in NAMES[1:]:
        shutil.copy(FIX / "old" / name, root / "docs" / "assets" / name)
    (root / "docs" / "assets" / "catalog-banner.png").write_bytes(PNG)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    shutil.copy(REPO / "scripts" / "release-check.py", root / "scripts" / "release-check.py")  # the gate reads the hygiene test beside it
    shutil.copy(REPO / "tests" / "test_public_hygiene.py", root / "tests" / "test_public_hygiene.py")
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "token.txt").write_text(TOKEN_TEXT)
    (secrets / "words.txt").write_text("zzprivate-word\n")
    (secrets / "chrome.exe").write_bytes(b"")
    return root


def env_for(root: Path) -> dict:
    secrets = root.parent / "secrets"
    return {"HF_TOKEN_PATH": str(secrets / "token.txt"), "HF_TOKEN": "must-not-pass", "HUGGING_FACE_HUB_TOKEN": "must-not-pass",
            "MODELROOM_RELEASE_PRIVATE_WORDS": str(secrets / "words.txt"), "MODELROOM_CHROME": str(secrets / "chrome.exe")}


def call(root: Path, *argv: str, runner: Runner | None = None, env: dict | None = None, capsys=None):
    import io

    out, err = io.StringIO(), io.StringIO()
    runner = runner if runner is not None else Runner()
    code = card.main(list(argv), root=root, run=runner, env=env_for(root) if env is None else env, out=out, err=err)
    return code, out.getvalue(), err.getvalue(), runner


def tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# --- numbers -------------------------------------------------------------------------------------------------------


def test_numbers_are_the_two_lists_and_the_size_in_kilobytes_rounded():
    assert card.numbers(catalog_bytes(231, 56, 171215)) == NEW
    assert card.numbers(catalog_bytes(80, 54, 65400)) == OLD
    assert card.numbers(catalog_bytes(3, 2, 171600)).kilobytes == 172


@pytest.mark.parametrize("data", [b"not json", b"[]", b'{"models": []}', b'{"publishers": []}', b'{"models": 3, "publishers": []}',
                                  b'{"models": [], "publishers": "x"}', b"\xff\xfe"])
def test_numbers_refuse_what_is_no_catalog(data):
    with pytest.raises(card.CardError):
        card.numbers(data)


# --- the card ------------------------------------------------------------------------------------------------------


def test_the_card_goes_from_the_old_numbers_to_the_new_and_back_byte_for_byte():
    old, new = golden("old", NAMES[0]).decode(), golden("new", NAMES[0]).decode()
    assert card.card_text(old, NEW) == new and card.card_text(new, OLD) == old
    assert card.card_text(new, NEW) == new and "·" in new


def test_the_card_refuses_a_pattern_that_does_not_hit_exactly_once():
    text = golden("new", NAMES[0]).decode()
    bold = next(line for line in text.splitlines() if line.startswith("**"))
    sentence = next(line for line in text.splitlines() if "Every morning a scan reads the" in line)
    for broken in (text.replace("publisher accounts watched", "publishers watched"), text.replace(" publisher accounts and the trending", " accounts and the trending")):
        with pytest.raises(card.CardError):
            card.card_text(broken, OLD)
    with pytest.raises(card.CardError):
        card.card_text(text + "\n" + bold, OLD)
    with pytest.raises(card.CardError):
        card.card_text(text + "\n" + sentence, OLD)


# --- the pictures --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES[1:])
def test_a_picture_goes_from_the_old_numbers_to_the_new_and_back_byte_for_byte(name):
    old, new = golden("old", name).decode(), golden("new", name).decode()
    assert card.svg_text(old, NEW) == new and card.svg_text(new, OLD) == old
    assert card.svg_text(new, NEW) == new


@pytest.mark.parametrize("name, font, step", [("catalog-banner.svg", 18, 10), ("catalog-social.svg", 19, 11)])
def test_a_digit_more_widens_the_first_pill_by_the_step_and_moves_the_next_ones(name, font, step):
    text = golden("new", name).decode()
    assert f'font-size="{font}"' in text and step == round(font * 0.57)
    wider = card.svg_text(text, card.CardNumbers(1000, 56, 171))
    assert ">1000 models<" in wider
    assert [int(w) for w in re.findall(r'<rect x="\d+" y="(?:238|410)" width="(\d+)"', wider)][:3] == [
        int(w) + step for w in re.findall(r'<rect x="\d+" y="(?:238|410)" width="(\d+)"', text)][:1] + [
        int(w) for w in re.findall(r'<rect x="\d+" y="(?:238|410)" width="(\d+)"', text)][1:3]
    again = card.svg_text(wider, card.CardNumbers(231, 56, 171))
    assert again == text


def test_a_digit_more_in_the_accounts_widens_the_second_pill_and_moves_the_third():
    text = golden("new", "catalog-banner.svg").decode()
    wider = card.svg_text(text, card.CardNumbers(231, 156, 171))
    before = [int(x) for x in re.findall(r'<rect x="(\d+)" y="238"', text)]
    after = [int(x) for x in re.findall(r'<rect x="(\d+)" y="238"', wider)]
    widths = [int(w) for w in re.findall(r'<rect x="\d+" y="238" width="(\d+)"', wider)]
    assert ">156 publishers watched<" in wider and after == [before[0], before[1], before[2] + 10] and widths[1] == 257
    assert card.svg_text(wider, card.CardNumbers(231, 56, 171)) == text


@pytest.mark.parametrize("mutate", [lambda t: t.replace(">231 models<", ">231 packages<"), lambda t: t.replace('<g class="chip"', '<g class="pill"'),
                                    lambda t: t.replace("</svg>", "<g class=\"chip\" font-size=\"18\"></g></svg>")])
def test_a_picture_refuses_a_pill_pattern_that_does_not_hit_exactly_once(mutate):
    with pytest.raises(card.CardError):
        card.svg_text(mutate(golden("new", "catalog-banner.svg").decode()), OLD)


# --- the check -----------------------------------------------------------------------------------------------------


def test_the_check_names_each_place_whose_number_is_not_the_catalogs(repo):
    code, out, err, _ = call(repo, "--check")
    lines = out.splitlines()
    assert code == 1 and err == "" and len(lines) == 8 and all(line.startswith("ALARM card: ") for line in lines)
    assert any("says 80, the catalog has 231" in line for line in lines) and any("says 54, the catalog has 56" in line for line in lines)
    assert any("says 65, the catalog has 171" in line for line in lines)


def test_the_check_is_silent_and_exits_0_when_every_place_agrees(repo):
    for name in NAMES:
        target = repo / "docs" / name if name.endswith(".md") else repo / "docs" / "assets" / name
        target.write_bytes(golden("new", name))
    assert call(repo, "--check")[:3] == (0, "", "")


def test_the_check_exits_2_when_the_structure_changed(repo):
    (repo / "docs" / "dataset-card.md").write_text("nothing of the old lines", encoding="utf-8")
    code, out, err, _ = call(repo, "--check")
    assert code == 2 and out == "" and err.startswith("error: ")
    (repo / "modelroom" / "catalog.json").write_bytes(b"{}")
    assert call(repo, "--check")[0] == 2


def test_the_check_writes_nothing_and_starts_nothing(repo):
    before = tree(repo)
    code, _, _, runner = call(repo, "--check")
    assert code == 1 and tree(repo) == before and runner.calls == []


# --- the write -----------------------------------------------------------------------------------------------------


def test_write_brings_the_three_files_to_the_goldens_and_a_second_run_leaves_them(repo):
    code, out, _, runner = call(repo, "--write")
    assert code == 0 and out.splitlines() == [f"written {p}" for p in ("docs/dataset-card.md", "docs/assets/catalog-banner.svg", "docs/assets/catalog-social.svg")]
    assert (repo / "docs" / "dataset-card.md").read_bytes() == golden("new", NAMES[0])
    assert (repo / "docs" / "assets" / "catalog-banner.svg").read_bytes() == golden("new", NAMES[1])
    assert (repo / "docs" / "assets" / "catalog-social.svg").read_bytes() == golden("new", NAMES[2])
    again = call(repo, "--write")
    assert again[0] == 0 and all(line.startswith("unchanged ") for line in again[1].splitlines()) and runner.calls == [] and call(repo, "--check")[:2] == (0, "")


def test_write_stops_before_any_file_when_one_pattern_fails(repo):
    (repo / "docs" / "assets" / "catalog-social.svg").write_text("<svg></svg>", encoding="utf-8")
    before = tree(repo)
    assert call(repo, "--write")[0] == 2 and tree(repo) == before


# --- the render ----------------------------------------------------------------------------------------------------


def test_render_runs_the_browser_headless_with_a_fresh_profile_each_time_and_writes_both_pictures(repo):
    code, out, _, runner = call(repo, "--render")
    assert code == 0 and len(runner.calls) == 2 and out.splitlines() == ["rendered docs/assets/catalog-banner.png", "rendered docs/assets/catalog-social.png"]
    (banner, env), (social, _) = runner.calls
    chrome = env_for(repo)["MODELROOM_CHROME"]
    assert banner[0] == chrome and banner[1:5] == ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1280,320"]
    assert social[4] == "--window-size=1280,640"
    assert banner[5].startswith("--screenshot=") and banner[6].startswith("--user-data-dir=") and banner[7] == (repo / "docs" / "assets" / "catalog-banner.svg").as_uri()
    assert banner[6] != social[6] and not Path(banner[6].split("=", 1)[1]).exists()
    assert (repo / "docs" / "assets" / "catalog-banner.png").read_bytes() == PNG and (repo / "docs" / "assets" / "catalog-social.png").read_bytes() == PNG
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env and TOKEN_TEXT not in repr(runner.calls)


def test_render_uses_the_standard_path_when_the_variable_is_not_set(tmp_path):
    runner, svg = Runner(), tmp_path / "x.svg"
    svg.write_bytes(golden("new", NAMES[1]))
    standard = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    card.render(svg, tmp_path / "x.png", runner, {}, isfile=lambda p: p == standard)
    assert runner.calls[0][0][0] == standard


def test_render_without_a_browser_says_so_and_writes_nothing(repo):
    env = {**env_for(repo), "MODELROOM_CHROME": str(repo / "no-such-chrome.exe")}
    before = tree(repo)
    code, _, err, runner = call(repo, "--render", env=env)
    assert code == 2 and "MODELROOM_CHROME" in err and runner.calls == [] and tree(repo) == before


def test_render_refuses_a_picture_of_10_kb_or_less(repo):
    runner = Runner(png=b"\x89PNG" + b"\0" * 100)
    before = tree(repo)
    code, _, err, _ = call(repo, "--render", runner=runner)
    assert code == 2 and "catalog-banner.png" in err and tree(repo) == before


def test_render_refuses_a_failed_browser_and_a_missing_screenshot(repo):
    chrome = env_for(repo)["MODELROOM_CHROME"]
    before = tree(repo)
    assert call(repo, "--render", runner=Runner(codes={chrome: 1}))[0] == 2 and tree(repo) == before

    class Silent(Runner):
        def __call__(self, args, env):
            self.calls.append((list(args), dict(env)))
            return 0

    assert call(repo, "--render", runner=Silent())[0] == 2 and tree(repo) == before


# --- the upload ----------------------------------------------------------------------------------------------------


def test_upload_is_one_process_with_the_card_and_the_banner_in_a_folder_of_its_own(repo):
    shutil.copy(FIX / "new" / NAMES[0], repo / "docs" / NAMES[0])
    code, out, err, runner = call(repo, "--upload")
    (args, child), = runner.calls
    assert code == 0 and err == "" and args[:5] == ["uvx", "--from", f"huggingface_hub=={P.HF_HUB_VERSION}", "hf", "upload"]
    assert args[5] == "jm-vis/modelroom-catalog" and args[7] == "." and args[8:10] == ["--repo-type", "dataset"] and args[10] == "--commit-message"
    assert args == P.upload_folder_args("jm-vis/modelroom-catalog", args[6], args[11]) and not Path(args[6]).exists()
    assert runner.folders == [{"README.md": golden("new", NAMES[0]), "catalog-banner.png": PNG}]
    assert child["HF_TOKEN_PATH"] == env_for(repo)["HF_TOKEN_PATH"] and "HF_TOKEN" not in child and "HUGGING_FACE_HUB_TOKEN" not in child
    assert TOKEN_TEXT not in repr(runner.calls) + out + err and "social" not in repr(runner.folders)


def test_upload_goes_to_the_named_dataset_and_the_probe_to_the_probe_dataset(repo):
    assert call(repo, "--upload", "--dataset", "someone/else")[3].calls[0][0][5] == "someone/else"
    assert call(repo, "--upload", "--probe")[3].calls[0][0][5] == "jm-vis/modelroom-catalog-probe"
    assert call(repo, "--upload", "--probe", "--dataset", "a/b")[0] == 2


def test_upload_refuses_a_private_word_in_the_card_before_any_process(repo):
    (repo / "docs" / "dataset-card.md").write_text("a line with zzprivate-word in it", encoding="utf-8")
    code, _, err, runner = call(repo, "--upload")
    assert code == 1 and runner.calls == [] and "zzprivate-word" not in err


@pytest.mark.parametrize("drop", ["MODELROOM_RELEASE_PRIVATE_WORDS", "HF_TOKEN_PATH"])
def test_upload_refuses_without_the_words_file_or_the_token_path(repo, drop):
    code, _, err, runner = call(repo, "--upload", env={k: v for k, v in env_for(repo).items() if k != drop})
    assert code == 1 and runner.calls == [] and drop in err and TOKEN_TEXT not in err


def test_upload_exits_1_when_the_process_fails(repo):
    code, _, err, runner = call(repo, "--upload", runner=Runner(codes={"uvx": 3}))
    assert code == 1 and "exit 3" in err and len(runner.calls) == 1


# --- the call ------------------------------------------------------------------------------------------------------


def test_write_render_and_upload_run_in_that_order_and_the_first_failure_stops(repo):
    code, out, _, runner = call(repo, "--write", "--render", "--upload")
    kinds = ["browser" if c[0][0] != "uvx" else "upload" for c in runner.calls]
    assert code == 0 and kinds == ["browser", "browser", "upload"] and out.splitlines()[0].startswith("written ")
    assert runner.folders[0]["README.md"] == golden("new", NAMES[0])
    stopped = call(repo, "--write", "--render", "--upload", env={**env_for(repo), "MODELROOM_CHROME": str(repo / "none.exe")})
    assert stopped[0] == 2 and stopped[3].calls == []


@pytest.mark.parametrize("argv", [(), ("--check", "--write"), ("--check", "--upload"), ("--dataset", "a/b"), ("--probe",), ("--write", "--probe"), ("--bogus",)])
def test_a_call_with_no_mode_or_a_wrong_mix_is_exit_2_and_touches_nothing(repo, argv):
    before = tree(repo)
    code, _, _, runner = call(repo, *argv)
    assert code == 2 and tree(repo) == before and runner.calls == []


# --- round 2 (review kac1) -----------------------------------------------------------------------------------------


def test_upload_reads_the_written_readme_back_and_refuses_other_bytes_without_a_process(repo):
    import io

    class Flips(card.DiskFiles):
        def read(self, path):
            data = super().read(path)
            return b"changed on the way" if Path(path).name == "README.md" and "catalog-card-" in str(path) else data

    runner, err = Runner(), io.StringIO()
    code = card.main(["--upload"], root=repo, files=Flips(), run=runner, env=env_for(repo), out=io.StringIO(), err=err)
    assert code == 1 and runner.calls == [] and "bytes" in err.getvalue()


def test_the_card_refuses_the_sentence_before_the_bold_line():
    text = golden("new", NAMES[0]).decode()
    sentence = next(line for line in text.splitlines() if "Every morning a scan reads the" in line)
    with pytest.raises(card.CardError):
        card.card_text(sentence + "\n" + text.replace(sentence, "", 1), OLD)


@pytest.mark.parametrize("label", ["80 models", "54 publishers watched"])
@pytest.mark.parametrize("name", NAMES[1:])
def test_a_picture_refuses_a_second_number_pill(name, label):
    text = golden("new", name).decode().replace(">CC BY 4.0<", f">{label}<")
    with pytest.raises(card.CardError):
        card.svg_text(text, OLD)


@pytest.mark.parametrize("where, mode", [(w, m) for m in ("--check", "--write", "--render") for w in
                                         ("docs/dataset-card.md", "docs/assets/catalog-banner.svg", "docs/assets/catalog-social.svg")
                                         if not (m == "--render" and w.endswith(".md"))])  # the render never reads the card
def test_broken_utf8_in_the_card_or_a_picture_is_exit_2_naming_the_file(repo, where, mode):
    (repo / where).write_bytes(b"\xff\xfe broken")
    before = tree(repo)
    code, _, err, runner = call(repo, mode)
    assert code == 2 and Path(where).name in err and "Traceback" not in err and (mode == "--render" or tree(repo) == before)


def test_upload_exits_1_when_the_upload_program_cannot_be_started(repo):
    def refuses(args, env):
        raise FileNotFoundError("no such program")

    code, _, err, _ = call(repo, "--upload", runner=refuses)
    assert code == 1 and err.startswith("failed: ")


def test_upload_exits_1_when_the_temporary_folder_cannot_be_made(repo, monkeypatch):
    def refuses(*args, **kwargs):
        raise PermissionError("no folder")

    monkeypatch.setattr(card.tempfile, "TemporaryDirectory", refuses)
    code, _, err, runner = call(repo, "--upload")
    assert code == 1 and err.startswith("failed: ") and runner.calls == []


@pytest.mark.parametrize("argv", [("--write", "--render"), ("--write", "--upload"), ("--render", "--upload")])
def test_a_pair_of_steps_is_exit_2_without_a_side_effect(repo, argv):
    before = tree(repo)
    code, _, _, runner = call(repo, *argv)
    assert code == 2 and tree(repo) == before and runner.calls == []


def test_one_model_is_written_as_1_models_and_a_template_with_1_model_is_refused():
    one = card.CardNumbers(1, 56, 171)
    assert "**1 models · 56" in card.card_text(golden("new", NAMES[0]).decode(), one)
    assert ">1 models<" in card.svg_text(golden("new", NAMES[1]).decode(), one)
    with pytest.raises(card.CardError):
        card.card_text(golden("new", NAMES[0]).decode().replace("231 models ·", "1 model ·"), NEW)
    with pytest.raises(card.CardError):
        card.svg_text(golden("new", NAMES[1]).decode().replace(">231 models<", ">1 model<"), NEW)
