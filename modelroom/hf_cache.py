"""Where vLLM keeps its models, and what of a package is already there (AP9-SV2-I, I2 and I3).

vLLM downloads a model into the Hugging Face cache when it first starts, so modelroom's disk rule
for a vLLM row reads the free space *there*, not where Ollama keeps its own (U11): the place is
`HF_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
`~/.cache/huggingface/hub` (huggingface_hub, "Environment variables"; an empty variable is unset,
as in the shell). `HF_CACHE_EXPRESSION` is the very same rule as one POSIX shell expression: the
Docker line of `install_vllm` mounts exactly that place and makes it first, so the download lands
where the disk was checked and "already there" was counted -- and no host path stands in a command.

A package is on disk when every start file (`tensor_quant.start_files`, one definition for the
download, the disk rule and this check) lies in `<cache>/models--<owner>--<name>/snapshots/<sha>/`
with the size the package declares. Size, never a digest: hashing several GB costs minutes. Only
`stat` is ever read, never a file's content, and no path is touched before every name is checked
(`SAFE_REPO_PATH_RE`, the repo against the Hub's name rule, the revision against 40 hex digits); a
link that leaves `models--<owner>--<name>` makes the answer unknown (`None`), as does a cache that
cannot be read. Everything touching the disk goes through `CacheProbes`, so a test passes its own.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Mapping, Sequence

from .contracts import Package, _HF_REPO_RE
from .quantization import package_identity_key
from .room import Disk, RoomProbes, free_disk_at, gb
from .tensor_quant import is_safe_repo_path, start_files

# The same rule as `hf_cache_candidates`, as the shell would resolve it: used inside double quotes by
# the Docker line, in the `mkdir` and in the bind mount, character for character.
HF_CACHE_EXPRESSION = "${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}"
_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def hf_cache_candidates(environ: Mapping[str, str], home: Path, system: str = "linux") -> list[Path]:
    """Where vLLM keeps its models: one place, the first of the four rules that applies.

    A list like `room.ollama_models_candidates` (the shape of `Runtime.model_place`), but never more
    than one: the Hugging Face rule picks the place, it does not leave two to choose from.
    """
    hub = environ.get("HF_HUB_CACHE")
    if hub:
        return [Path(hub)]
    hf_home = environ.get("HF_HOME")
    if hf_home:
        return [Path(hf_home) / "hub"]
    xdg = environ.get("XDG_CACHE_HOME")
    if xdg:
        return [Path(xdg) / "huggingface" / "hub"]
    return [home / ".cache" / "huggingface" / "hub"]


def cache_is_relative(environ: Mapping[str, str], system: str = "linux") -> bool:
    """Whether the variable that decides the place (the effective one, in the order above) is a
    relative path: `docker --mount type=bind` refuses it, so the Docker line is left out then (I1)."""
    effective = environ.get("HF_HUB_CACHE") or environ.get("HF_HOME") or environ.get("XDG_CACHE_HOME")
    if not effective:
        return False
    path = PureWindowsPath(effective) if system == "win32" else PurePosixPath(effective)
    return not path.is_absolute()


def _real_size(path: str) -> int:
    return os.stat(path).st_size


@dataclass(frozen=True)
class CacheProbes:
    """Everything the check touches: where a path really leads (`realpath`, links followed) and how
    large the file there is (`size`, `stat` only). A test passes its own."""

    realpath: Callable[[str], str] = os.path.realpath
    size: Callable[[str], int] = _real_size


@dataclass(frozen=True)
class CacheCheck:
    """What one package has in the cache: `on_disk` (`True`: every start file, `False`: not all of
    them, `None`: not known), and `missing_bytes`, the bytes the first start still downloads -- `0`
    when it is all there, `None` where nobody can say (unknown, or a missing start file whose declared
    size is 0: the disk rule then counts the whole download)."""

    on_disk: bool | None
    missing_bytes: int | None


_UNKNOWN = CacheCheck(None, None)


def _folder_name(package: Package) -> str | None:
    """`models--<owner>--<name>` for a package whose repo and revision are safe to build a path from."""
    repo, revision = package.repo, package.revision
    if not isinstance(repo, str) or not repo.isascii() or not _HF_REPO_RE.fullmatch(repo):
        return None
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        return None
    owner, name = repo.split("/", 1)
    return f"models--{owner}--{name}"


def _inside(base: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(base), os.path.normcase(path)]) == os.path.normcase(base)
    except ValueError:  # another drive, or one path absolute and the other not
        return False


def check_package(package: Package, cache: Path, probes: CacheProbes | None = None) -> CacheCheck:
    """`CacheCheck` of `package` in the cache folder `cache`: its start files against the snapshot folder.

    Every name and the revision are checked first, and every resolved path must lie inside
    `models--<owner>--<name>` before one size is read; then each start file is `stat`ed.
    """
    probes = probes if probes is not None else CacheProbes()
    folder = _folder_name(package)
    files = start_files(package.files)
    if folder is None or not files or not all(is_safe_repo_path(file.name) for file in files):
        return _UNKNOWN
    base = os.path.join(str(cache), folder)
    real_base = probes.realpath(base)
    paths = []
    for file in files:
        path = os.path.join(base, "snapshots", package.revision, *file.name.split("/"))
        if not _inside(real_base, probes.realpath(path)):
            return _UNKNOWN
        paths.append(path)
    missing, unsayable = 0, False
    for file, path in zip(files, paths):
        try:
            found: int | None = probes.size(path)
        except (FileNotFoundError, NotADirectoryError):
            found = None
        except OSError:
            return _UNKNOWN  # a cache that cannot be read says nothing
        if found == file.size_bytes:
            continue
        if file.size_bytes == 0:
            unsayable = True  # declared nothing: how much is there to fetch is not known
        else:
            missing += file.size_bytes
    if missing == 0 and not unsayable:
        return CacheCheck(True, 0)
    return CacheCheck(False, None if unsayable else missing)


def on_disk(package: Package, cache: Path, probes: CacheProbes | None = None) -> bool | None:
    """Whether `package` is on disk (`size checked`): `True` all start files, `False` not all or no
    cache, `None` where that cannot be told (an unsafe name, a link out of the repo folder, an unreadable cache)."""
    return check_package(package, cache, probes).on_disk


def _is_tensor(package: Package) -> bool:
    return package.source == "huggingface" and package.format == "tensor"


def _download_bytes(package: Package) -> int:
    return sum(file.size_bytes for file in start_files(package.files))


def vllm_disk(room: RoomProbes, packages: Sequence[Package] = ()) -> Disk:
    """The free disk where vLLM keeps its models, read now, with what the disk rule needs of each of
    `packages`: `Disk.held` the packages already there, `Disk.needs` the bytes still to fetch of each
    (the missing start files; the whole download where that cannot be told), `Disk.relative` whether a
    cache variable of the machine is relative. Measured at the nearest folder that exists, like Ollama's place."""
    place = hf_cache_candidates(room.environ, room.home, room.system)[0]
    base = free_disk_at(room, place)
    held: set[tuple[str, str, str]] = set()
    needs: dict[tuple[str, str, str], int] = {}
    for package in packages:
        if not _is_tensor(package):
            continue
        identity, check = package_identity_key(package), check_package(package, place)
        if check.on_disk is True:
            held.add(identity)
        needs[identity] = check.missing_bytes if check.missing_bytes is not None else _download_bytes(package)
    return replace(base, held=frozenset(held), needs=needs, relative=cache_is_relative(room.environ, room.system))


def shared_disk_words(disk: Disk) -> str:
    """`disk 41.2 GB free on C: (Ollama and vLLM models)`: the one piece of a machine whose two places
    are on one drive (I2)."""
    return f"disk {gb(disk.free_bytes)} free on {disk.drive} (Ollama and vLLM models)"


def vllm_disk_words(disk: Disk) -> str:
    """`disk 41.2 GB free on D: (vLLM models)`: vLLM's own place, on another drive than Ollama's or alone."""
    return f"disk {gb(disk.free_bytes)} free on {disk.drive} (vLLM models)"


__all__ = [
    "HF_CACHE_EXPRESSION",
    "CacheCheck",
    "CacheProbes",
    "cache_is_relative",
    "check_package",
    "hf_cache_candidates",
    "on_disk",
    "shared_disk_words",
    "vllm_disk",
    "vllm_disk_words",
]
