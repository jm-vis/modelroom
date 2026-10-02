"""Where vLLM keeps its models, what is "already there", and what the disk rule says (AP9-SV2-I, I2 and I3).

The place is the Hugging Face cache (`hf_cache.hf_cache_candidates`); the disk rule is the one of
AP9-RM U11 read at that place for vLLM's rows and at Ollama's place for Ollama's (`disk_fit`). A
package is "on disk" when every start file (P2) lies in the snapshot folder with the size the package
declares -- size, never a digest (I3). Every test imports what AP9-SV2-I added inside itself.
"""

from __future__ import annotations

import dataclasses
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelroom.contracts import PackageFile
from modelroom.room import GIB, Disk, LiveRoom
from modelroom.screen import plain

from fixture_room import fixed_room
from test_golden_vllm import FIXTURES, _base_model, _gguf_package, _profile, _render
from test_install_vllm import REVISION, _package

SHARD_1, SHARD_2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"


def _regular(size: int):
    """What `os.stat` says of a regular file of `size` bytes."""
    return SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_size=size)
FOLDER = "models--Qwen--Qwen3-8B-AWQ"


def _files(tokenizer: int = 50) -> list[PackageFile]:
    """Two shards, a tokenizer and two small configs (start files), and a README (not one)."""
    return [
        PackageFile(name=SHARD_1, role="weights_shard", size_bytes=1000, digest=None),
        PackageFile(name=SHARD_2, role="weights_shard", size_bytes=1200, digest=None),
        PackageFile(name="tokenizer.json", role="other", size_bytes=tokenizer, digest=None),
        PackageFile(name="config.json", role="other", size_bytes=30, digest=None),
        PackageFile(name="README.md", role="other", size_bytes=99_999, digest=None),
    ]


def _small(files: list[PackageFile] | None = None):
    return _package(files=files if files is not None else _files())


def _snapshot(hub: Path, revision: str = REVISION) -> Path:
    return hub / FOLDER / "snapshots" / revision


def _fill(hub: Path, package, *, skip=(), sizes: dict[str, int] | None = None) -> Path:
    """The cache of `package` under `hub`: every file of the package but `skip`, each as large as the
    package declares (or as `sizes` says). The README is never written: it is no start file."""
    snapshot = _snapshot(hub, package.revision)
    snapshot.mkdir(parents=True, exist_ok=True)
    for file in package.files:
        if file.name in skip or file.name == "README.md":
            continue
        size = (sizes or {}).get(file.name, file.size_bytes)
        (snapshot / file.name).write_bytes(b"x" * size)
    return hub


# --- I2: the place -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({"HF_HUB_CACHE": "/a/hub", "HF_HOME": "/b", "XDG_CACHE_HOME": "/c"}, Path("/a/hub")),
        ({"HF_HOME": "/b", "XDG_CACHE_HOME": "/c"}, Path("/b/hub")),
        ({"XDG_CACHE_HOME": "/c"}, Path("/c/huggingface/hub")),
        ({}, Path("/srv/user/.cache/huggingface/hub")),
        ({"HF_HUB_CACHE": "", "HF_HOME": "", "XDG_CACHE_HOME": ""}, Path("/srv/user/.cache/huggingface/hub")),
    ],
    ids=["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "none", "all empty"],
)
def test_each_hf_cache_variable_in_its_order(environ, expected):
    from modelroom.hf_cache import hf_cache_candidates

    assert hf_cache_candidates(environ, Path("/srv/user"), "linux") == [expected]


def test_the_cache_is_measured_at_the_nearest_existing_parent(tmp_path: Path):
    from modelroom.hf_cache import vllm_disk

    asked: list[Path] = []

    def usage(path: Path) -> int:
        asked.append(path)
        return 5 * GIB

    hub = tmp_path / "not" / "yet" / "hub"
    room = dataclasses.replace(
        fixed_room(environ={"HF_HUB_CACHE": str(hub)}), exists=lambda path: path.exists(), resolve=lambda path: path, disk_usage=usage
    )

    disk = vllm_disk(room, [])

    assert asked == [tmp_path] and disk.known and disk.free_bytes == 5 * GIB and disk.relative is False


def test_a_relative_cache_is_flagged_on_the_disk_read():
    from modelroom.hf_cache import vllm_disk

    disk = vllm_disk(fixed_room(environ={"HF_HUB_CACHE": "hf-cache"}), [])

    assert disk.relative is True


# --- I3: on disk --------------------------------------------------------------------------------------------


def test_all_files_with_their_sizes_are_on_disk(tmp_path: Path):
    from modelroom.hf_cache import check_package, on_disk

    package = _small()
    hub = _fill(tmp_path / "hub", package)

    assert on_disk(package, hub) is True
    assert check_package(package, hub).missing_bytes == 0  # the README is no start file, and not missing


def test_a_missing_shard_is_not_on_disk(tmp_path: Path):
    from modelroom.hf_cache import check_package, on_disk

    package = _small()
    hub = _fill(tmp_path / "hub", package, skip=(SHARD_2,))

    assert on_disk(package, hub) is False
    assert check_package(package, hub).missing_bytes == 1200


def test_a_wrong_size_is_not_on_disk(tmp_path: Path):
    from modelroom.hf_cache import check_package, on_disk

    package = _small()
    hub = _fill(tmp_path / "hub", package, sizes={SHARD_1: 999})

    assert on_disk(package, hub) is False
    assert check_package(package, hub).missing_bytes == 1000  # the declared size is fetched again


def test_a_missing_cache_is_not_on_disk(tmp_path: Path):
    from modelroom.hf_cache import check_package, on_disk

    package = _small()

    assert on_disk(package, tmp_path / "nothing") is False
    assert check_package(package, tmp_path / "nothing").missing_bytes == 1000 + 1200 + 50 + 30


def test_an_unreadable_cache_is_unknown(tmp_path: Path):
    from modelroom.hf_cache import CacheProbes, check_package, on_disk

    package = _small()
    hub = _fill(tmp_path / "hub", package)

    def denied(path: str):
        raise PermissionError(path)

    probes = CacheProbes(stat=denied)

    assert on_disk(package, hub, probes) is None
    assert check_package(package, hub, probes).missing_bytes is None


def test_missing_tokenizer_is_not_on_disk_and_the_missing_bytes_meet_the_disk_rule(tmp_path: Path):
    """Codex r1-6: the shards are there, the tokenizer (7 000 000 bytes declared) is not: the package is
    not on disk, and the disk rule counts those bytes plus the margin -- one byte short marks the row."""
    from modelroom.hf_cache import check_package, vllm_disk

    package = _small(_files(tokenizer=7_000_000))
    hub = _fill(tmp_path / "hub", package, skip=("tokenizer.json",))
    assert check_package(package, hub).missing_bytes == 7_000_000

    for free, marked in ((GIB + 7_000_000 - 1, True), (GIB + 7_000_000, False)):
        room = fixed_room(free_disk_bytes=free, environ={"HF_HUB_CACHE": str(hub)})
        vllm = vllm_disk(room, [package])
        assert _vllm_row_marked(package, vllm) is marked, free


def test_complete_start_files_skip_the_disk_rule(tmp_path: Path):
    """Every start file there, nothing free at all: the package needs nothing and its row is not marked."""
    from modelroom.hf_cache import vllm_disk

    package = _small()
    hub = _fill(tmp_path / "hub", package)
    room = fixed_room(free_disk_bytes=1, environ={"HF_HUB_CACHE": str(hub)})

    vllm = vllm_disk(room, [package])

    assert vllm.held and _vllm_row_marked(package, vllm) is False


def test_on_disk_refuses_unsafe_names_without_touching_files(tmp_path: Path):
    from modelroom.hf_cache import CacheProbes, on_disk

    def touched(path: str):
        raise AssertionError(f"a file was touched: {path}")

    probes = CacheProbes(realpath=touched, stat=touched)
    for name in ("/etc/x.safetensors", "../x.safetensors", "a/../../x.json", "..\\x.json", "C:\\x.json", "\\\\host\\share\\x.json", "a:b.json", "a/b/c/d/e.json"):
        package = _small([PackageFile(name=name, role="weights", size_bytes=1, digest=None)])
        assert on_disk(package, tmp_path / "hub", probes) is None, name
    for update in ({"repo": "a/b;rm"}, {"repo": "ab"}, {"revision": "main"}, {"revision": "A" * 40}, {"revision": None}, {"repo": None}):
        assert on_disk(_small().model_copy(update=update), tmp_path / "hub", probes) is None, update


def test_a_link_leaving_the_repo_cache_is_unknown(tmp_path: Path):
    """Codex r1-2: a start file whose resolved path lies outside `models--<owner>--<name>` (another repo, a
    folder next to the cache, the drive) is unknown, and its size is never read."""
    from modelroom.hf_cache import CacheProbes, on_disk

    package = _small()
    hub = _fill(tmp_path / "hub", package)
    other = hub / "models--Other--Repo" / "blobs" / "abc"
    outside = tmp_path / "outside" / "secret"
    sized: list[str] = []
    for target in (other, outside, Path("/etc/passwd"), Path("C:/Windows/system.ini")):
        probes = CacheProbes(realpath=lambda path, target=target: str(target) if path.endswith(SHARD_1) else os.path.realpath(path), stat=lambda path: sized.append(path) or _regular(1))
        assert on_disk(package, hub, probes) is None, target
    assert sized == []


def test_a_real_blob_link_inside_the_repo_folder_counts(tmp_path: Path):
    """The legitimate Hugging Face layout, `snapshots/<sha>/<file>` -> `../../blobs/<hash>`, is on disk;
    a real link to another repo's blob is unknown."""
    from modelroom.hf_cache import on_disk

    package = _small()
    hub = _fill(tmp_path / "hub", package)
    blobs = hub / FOLDER / "blobs"
    blobs.mkdir()
    shard = _snapshot(hub) / SHARD_1
    (blobs / "h1").write_bytes(shard.read_bytes())
    shard.unlink()
    try:
        os.symlink(os.path.join("..", "..", "blobs", "h1"), shard)
    except (OSError, NotImplementedError):
        pytest.skip("this machine may not create symbolic links")
    assert on_disk(package, hub) is True

    foreign = hub / "models--Other--Repo" / "blobs"
    foreign.mkdir(parents=True)
    (foreign / "h2").write_bytes(b"x" * 1000)
    shard.unlink()
    os.symlink(str(foreign / "h2"), shard)
    assert on_disk(package, hub) is None


def test_a_repo_folder_that_is_a_link_leads_nowhere(tmp_path: Path):
    """Codex r1-2 (R1-2): `models--<owner>--<name>` itself is a link to another repo's folder, or to a place
    outside the cache: the answer is unknown and no `stat` is made. A real folder with blob links stays valid."""
    from modelroom.hf_cache import CacheProbes, on_disk

    package = _small()
    foreign = _fill(tmp_path / "other-hub", package)
    outside = tmp_path / "outside"
    outside.mkdir()
    hub = tmp_path / "hub"
    hub.mkdir()
    stated: list[str] = []
    probes = CacheProbes(stat=lambda path: stated.append(path) or _regular(1))
    for target in (foreign / FOLDER, outside):
        try:
            os.symlink(target, hub / FOLDER, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("this machine may not create symbolic links")
        assert on_disk(package, hub, probes) is None, target
        (hub / FOLDER).unlink() if (hub / FOLDER).is_symlink() else None
    assert stated == []
    real = _fill(tmp_path / "real-hub", package)
    assert on_disk(package, real) is True


def test_the_repo_folder_anchor_is_the_resolved_cache_and_the_folder_name(tmp_path: Path):
    """R1-2 without a symbolic link: whatever `realpath` says of the repo folder must be the folder name below
    the resolved cache, else nothing is read."""
    from modelroom.hf_cache import CacheProbes, on_disk

    package = _small()
    hub = _fill(tmp_path / "hub", package)
    stated: list[str] = []
    moved = str(tmp_path / "elsewhere" / FOLDER)
    probes = CacheProbes(
        realpath=lambda path: moved + path[len(str(hub / FOLDER)):] if path.startswith(str(hub / FOLDER)) else os.path.realpath(path),
        stat=lambda path: stated.append(path) or _regular(1),
    )
    assert on_disk(package, hub, probes) is None
    assert stated == []


def test_the_size_is_read_at_the_resolved_path_never_the_original(tmp_path: Path):
    """R1-3: the one `stat` per start file is made at the path that passed the boundary check."""
    from modelroom.hf_cache import CacheProbes, check_package

    package = _small()
    hub = tmp_path / "hub"
    base = os.path.realpath(hub / FOLDER)
    seen: list[str] = []

    def realpath(path: str) -> str:
        real = os.path.realpath(path)
        marker = os.path.join(base, "snapshots", package.revision)
        return os.path.join(base, "blobs", "h-" + os.path.basename(real)) if real.startswith(marker) else real

    sizes = {file.name: file.size_bytes for file in package.files}

    def stat_(path: str):
        seen.append(path)
        return _regular(sizes[os.path.basename(path)[2:]])

    result = check_package(package, hub, CacheProbes(realpath=realpath, stat=stat_))

    assert result.on_disk is True
    assert len(seen) == 4 and all(os.path.join(base, "blobs") in path for path in seen)
    assert not any("snapshots" in path for path in seen)


def test_a_start_file_that_is_no_regular_file_is_not_there(tmp_path: Path):
    """R1-4: a folder (or a FIFO) in place of a size-0 `tokenizer.json` is not "on disk"."""
    from modelroom.hf_cache import check_package, on_disk

    package = _small(_files(tokenizer=0))
    hub = _fill(tmp_path / "hub", package, skip=("tokenizer.json",))
    assert on_disk(package, hub) is False  # missing: baseline
    (_snapshot(hub) / "tokenizer.json").mkdir()
    assert on_disk(package, hub) is False
    assert check_package(package, hub).missing_bytes is None
    if hasattr(os, "mkfifo"):
        (_snapshot(hub) / "tokenizer.json").rmdir()
        os.mkfifo(_snapshot(hub) / "tokenizer.json")
        assert on_disk(package, hub) is False
    # the same file as a regular one of size 0 is there
    target = _snapshot(hub) / "tokenizer.json"
    if target.is_dir():
        target.rmdir()
    elif target.exists():
        target.unlink()
    target.write_bytes(b"")
    assert on_disk(package, hub) is True


def test_local_zero_size_tokenizer_missing_is_null_not_on_disk(tmp_path: Path):
    """Codex r3-6: every weight is there, `tokenizer.json` declares size 0 and is missing: nobody can say
    how much is missing -- `missing_bytes` is `None`, the package is not on disk, and the disk rule counts the
    whole download (the sum of the start files)."""
    from modelroom.hf_cache import check_package, vllm_disk
    from modelroom.quantization import package_identity_key

    package = _small(_files(tokenizer=0))
    hub = _fill(tmp_path / "hub", package, skip=("tokenizer.json",))

    result = check_package(package, hub)
    assert result.on_disk is False and result.missing_bytes is None

    whole = 1000 + 1200 + 0 + 30
    vllm = vllm_disk(fixed_room(free_disk_bytes=GIB + whole - 1, environ={"HF_HUB_CACHE": str(hub)}), [package])
    assert vllm.needs[package_identity_key(package)] == whole and not vllm.held
    assert _vllm_row_marked(package, vllm) is True


def test_zero_size_start_file_of_wrong_size_is_null(tmp_path: Path):
    from modelroom.hf_cache import check_package

    package = _small(_files(tokenizer=0))
    hub = _fill(tmp_path / "hub", package, sizes={"tokenizer.json": 5})  # declared 0, but 5 bytes lie there

    result = check_package(package, hub)

    assert result.on_disk is False and result.missing_bytes is None


# --- I2: the disk rule per runtime (U11 at the right place) ------------------------------------------------


def _vllm_row_marked(package, vllm: Disk) -> bool:
    """Whether the rank-1 row of a vLLM-only machine is marked over the disk, through the whole terminal view."""
    import json

    from modelroom.profile import HardwareProfile
    from modelroom.views import RunView, terminal_blocks

    data = json.loads((FIXTURES / "profiles_v4" / "four_l4_server_vllm_only.json").read_text(encoding="utf-8"))
    document = _render(HardwareProfile.model_validate(data), [_base_model()], [package])
    live = {"server": LiveRoom(disk=Disk(10**13, "C:", vllm=vllm))}
    lines = [plain(line).strip() for line in terminal_blocks(document, view=RunView(install_for="server", live=live, packages=[package]))]
    row = next(line for line in lines if line.startswith("1 "))
    return row.endswith("×")


def _big_gguf_and_awq():
    """One GGUF row and one tensor row of the same base model, for the two places of a machine with both."""
    return [_gguf_package(), _package()]


def _both_view(ollama: Disk, vllm: Disk | None):
    from modelroom.views import RunView, terminal_blocks

    packages = _big_gguf_and_awq()
    document = _render(_profile(), [_base_model()], packages)
    live = {"server": LiveRoom(disk=dataclasses.replace(ollama, vllm=vllm))}
    return [plain(line).strip() for line in terminal_blocks(document, view=RunView(install_for="server", live=live, packages=packages))]


def test_too_little_disk_at_the_vllm_place_marks_the_row():
    """U11's form at vLLM's own place: the download is 6 098 581 864 bytes (5.7 GB) of weights; 5.0 GB free
    on D: marks the vLLM row (`×` behind its download, the `disk` line, `no package fits` for its own
    install line) while Ollama's row, on a roomy C:, is untouched."""
    text = _both_view(Disk(100 * GIB, "C:"), Disk(5 * GIB, "D:"))

    rows = {line.split()[0]: line for line in text if line[:2] in ("1 ", "2 ")}
    assert rows["2"].endswith("5.7 GB ×") and not rows["1"].endswith("×")
    assert any(line.startswith("disk      × #2 needs more than the 5.0 GB free on D:") for line in text)
    assert "install   no package fits the 5.0 GB free on D:; free space first" in text
    assert any(line.startswith("install   #1  ollama pull ") for line in text)


def test_the_pull_list_locks_a_vllm_row_with_the_u11_words():
    from modelroom.pull_choice import pull_rows

    packages = _big_gguf_and_awq()
    document = _render(_profile(), [_base_model()], packages)
    disk = dataclasses.replace(Disk(100 * GIB, "C:"), vllm=Disk(5 * GIB, "D:"))

    rows = {row.rank: row for row in pull_rows(document, "server", disk=disk, packages=packages)}

    assert rows[1].locked is None and rows[2].locked == "needs 5.7 GB plus 1 GB room, 5.0 GB free" and rows[2].over_disk


def test_the_ollama_place_decides_gguf_rows_only():
    """The reverse: Ollama's place has 1 GB, vLLM's 100 GB: the GGUF row is marked, the tensor row is not."""
    text = _both_view(Disk(1 * GIB, "C:"), Disk(100 * GIB, "D:"))

    rows = {line.split()[0]: line for line in text if line[:2] in ("1 ", "2 ")}
    assert rows["1"].endswith("×") and not rows["2"].endswith("×")
    assert any(line.startswith("disk      × #1 needs more than the 1.0 GB free on C:") for line in text)


def test_a_vllm_row_is_never_judged_at_an_unknown_vllm_place():
    text = _both_view(Disk(1 * GIB, "C:"), None)

    assert not any(line.endswith("×") for line in text if line[:2] == "2 ")


def _room_text(ollama: Disk, vllm: Disk | None) -> str:
    return " ".join(line for line in _both_view(ollama, vllm) if line.startswith(("room", "disk ")) or "free on" in line)


def test_same_drive_gives_one_disk_piece():
    """Both places on `C:`: one piece, `disk 41.2 GB free on C: (Ollama and vLLM models)`; on two drives, two."""
    free = 44_236_955_648  # 41.2 GB
    same = _room_text(Disk(free, "C:"), Disk(free, "C:"))
    assert same.count("disk 41.2 GB free on C: (Ollama and vLLM models)") == 1 and "(Ollama models)" not in same

    apart = _room_text(Disk(free, "C:"), Disk(2 * free, "D:"))
    assert "disk 41.2 GB free on C: (Ollama models)" in apart and "disk 82.4 GB free on D: (vLLM models)" in apart
    assert "Ollama and vLLM" not in apart


def test_a_machine_without_vllm_keeps_its_disk_piece_as_it_was():
    text = _both_view_without_vllm(Disk(44_236_955_648, "C:"))

    assert "disk 41.2 GB free on C: (Ollama models)" in text and "vLLM models" not in text


def _both_view_without_vllm(ollama: Disk) -> str:
    from modelroom.views import RunView, terminal_blocks

    document = _render(_profile().model_copy(update={"runtimes": ["ollama"]}), [_base_model()], [_gguf_package()])
    live = {"server": LiveRoom(disk=dataclasses.replace(ollama, vllm=Disk(1 * GIB, "D:")))}
    return " ".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", live=live)))


def test_a_vllm_only_machine_names_its_own_disk():
    import json

    from modelroom.profile import HardwareProfile
    from modelroom.views import RunView, terminal_blocks

    data = json.loads((FIXTURES / "profiles_v4" / "four_l4_server_vllm_only.json").read_text(encoding="utf-8"))
    document = _render(HardwareProfile.model_validate(data), [_base_model()], [_package()])
    live = {"server": LiveRoom(disk=Disk(10 * GIB, "C:", vllm=Disk(44_236_955_648, "D:")))}

    text = " ".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", live=live, packages=[_package()])))

    assert "disk 41.2 GB free on D: (vLLM models)" in text and "Ollama" not in text
