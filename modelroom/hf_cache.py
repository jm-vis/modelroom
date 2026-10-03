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
import stat
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

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


@dataclass(frozen=True)
class CacheProbes:
    """Everything the check touches: where a path really leads (`realpath`, links followed) and what
    is there (`stat`: its `st_mode` and `st_size`, metadata only, never the content). `stat` is asked
    of the resolved path that passed the boundary check, once per file. A test passes its own."""

    realpath: Callable[[str], str] = os.path.realpath
    stat: Callable[[str], Any] = os.stat


@dataclass(frozen=True)
class CacheCheck:
    """What one package has in the cache: `on_disk` (`True`: every start file, `False`: not all of
    them, `None`: not known), and `missing_bytes`, the bytes the first start still downloads -- `0`
    when it is all there, `None` where nobody can say (unknown, or a missing start file whose declared
    size is 0: the disk rule then counts the whole download)."""

    on_disk: bool | None
    missing_bytes: int | None


_UNKNOWN = CacheCheck(None, None)


def _folder_name(repo: object, revision: object) -> str | None:
    """`models--<owner>--<name>` for a repo and revision that are safe to build a path from."""
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


@dataclass(frozen=True)
class _StartRef:
    """What the check reads of a package: its repo, its revision and its start files, nothing else
    (the shape of `remote_vllm.package_ref`, which is what crosses the wire to a server)."""

    repo: object
    revision: object
    files: Sequence[Any]


def _inspect(ref: Any, cache: Path, probes: CacheProbes | None) -> CacheCheck:
    """The one place the rules of the cache check live (I3, Rev 3, Rev 4), for a package of this
    computer and for the reference a server is sent alike. `ref` has `repo`, `revision` and `files`,
    each with `name` and `size_bytes`; every file of it is a start file.

    Every name and the revision are checked first. The repo folder itself must be where the cache puts
    it (`realpath(cache)/models--<owner>--<name>`: a link that moves the folder away is not followed),
    and every resolved path must lie inside it -- or inside the cache's own blob store
    `realpath(cache)/blobs/`, where huggingface_hub 1.x keeps every large file it fetched with Xet,
    shared by all repos of the cache (`snapshots/<sha>/<file>` -> `../../blobs/<etag>` ->
    `<cache>/blobs/<xx>/<sha256>`; seen at a rented server on 2026-10-03) -- before one `stat` is
    made; then each start file is `stat`ed at that resolved path and counts only as a regular file of
    the declared size. Another repo's folder, a place beside the cache, another drive: not inside.
    """
    probes = probes if probes is not None else CacheProbes()
    folder = _folder_name(ref.repo, ref.revision)
    files = list(ref.files)
    if folder is None or not files or not all(is_safe_repo_path(file.name) for file in files):
        return _UNKNOWN
    base = os.path.join(str(cache), folder)
    real_base = probes.realpath(base)
    real_cache = probes.realpath(str(cache))
    if os.path.normcase(real_base) != os.path.normcase(os.path.join(real_cache, folder)):
        return _UNKNOWN  # the repo folder is a link that leads elsewhere: nothing below it is touched
    store = os.path.join(real_cache, "blobs")
    resolved = []
    for file in files:
        path = os.path.join(base, "snapshots", ref.revision, *file.name.split("/"))
        real = probes.realpath(path)
        if not (_inside(real_base, real) or _inside(store, real)):
            return _UNKNOWN
        resolved.append(real)
    missing, unsayable = 0, False
    for file, real in zip(files, resolved):
        try:
            info = probes.stat(real)
            found: int | None = info.st_size if stat.S_ISREG(info.st_mode) else None  # a FIFO, a folder: not there
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


def missing_bytes(ref: Any, cache: Path, probes: CacheProbes | None = None) -> int | None:
    """The bytes the first start of `ref` still has to download: `0` only where every start file is
    there with its declared size (a missing file of declared size 0 is never `0`), the sum of the
    missing ones otherwise, `None` where nobody can say (an unsafe name, a link that leaves the repo
    folder and the cache's blob store, an unreadable cache, a missing start file of declared size 0).

    The one function of both ends (AP9-SV2-R, R2, Codex r2-6, r3-6): this computer's disk rule reads it
    for its own packages (`vllm_disk`), a server reads it for the references it was sent
    (`remote_server.vllm_result`), so the rules above are in one place."""
    return _inspect(ref, cache, probes).missing_bytes


def check_package(package: Package, cache: Path, probes: CacheProbes | None = None) -> CacheCheck:
    """`CacheCheck` of `package` in the cache folder `cache`: its start files against the snapshot folder."""
    return _inspect(_start_ref(package), cache, probes)


def _start_ref(package: Package) -> _StartRef:
    return _StartRef(package.repo, package.revision, start_files(package.files))


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
        identity, missing = package_identity_key(package), missing_bytes(_start_ref(package), place)
        if missing == 0:  # `0` is only ever said of a package whose start files are all there
            held.add(identity)
        needs[identity] = missing if missing is not None else _download_bytes(package)
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
    "missing_bytes",
    "on_disk",
    "shared_disk_words",
    "vllm_disk",
    "vllm_disk_words",
]
