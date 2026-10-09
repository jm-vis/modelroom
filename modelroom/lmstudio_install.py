"""What the LM Studio entry installs and holds: its command, its model folder, "already there", its disk.

modelroom prints `lms get --gguf https://huggingface.co/<owner>/<repo>@<quant>` and never runs it (G7):
the URL, because `lms get <owner>/<name>` names an artifact of the LM Studio Hub (`lms get --help`). The
reader copies the line, and `lms` stays a program modelroom does not start (CONTRACTS.md, "Local servers and the vLLM load
test", LM Studio). Every value that reaches a line is checked here and nowhere else: the repository
against the Hub's name rule and plain ASCII, the stage against the stages of `QUANT_ORDER` that are GGUF
and not `UD-` (`LMS_QUANTS`), the file against `SAFE_REPO_PATH_RE`. A package with more than one weight
file, a file in a subfolder or a stage outside that list has no command and says why
(`lms_reason`); the row stays in the ranking with its fit.

LM Studio keeps its models in `~/.lmstudio/models/<owner>/<repo>/<file>` (the standard folder; a changed
folder is not read). A package is on disk when its one weight file lies there with the size the catalog
revision declares: `stat` only, never the content, and no path is touched before every name is checked.
`lmstudio_disk` hangs the entry on the disk of a guided run: the runtimes of the machine, so that the
Markdown file judges a row as the terminal does, and the free space at LM Studio's folder (F6 a).

The head imports are the small ones of `hf_cache.py`; `runtimes.py` imports this module when it loads, so
the imports back into it (and into `importer`) are made at the call.
"""

from __future__ import annotations

import os
import re
import stat
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from .contracts import Package, _HF_REPO_RE
from .hf_cache import CacheProbes
from .lmstudio_server import lms_call
from .quantization import QUANT_ORDER, package_identity_key
from .room import Disk, RoomProbes, free_disk_at
from .tensor_quant import SAFE_REPO_PATH_RE

if TYPE_CHECKING:  # pragma: no cover - the configuration is passed in, never read here
    from .config import Configuration

_TENSOR_STAGES = frozenset({"W4A16", "AWQ", "GPTQ-4", "GPTQ-8", "W8A8", "FP8"})
# The stages `lms get --gguf` is allowed here: the GGUF stages of `QUANT_ORDER`, the tensor labels (they
# are vLLM's) and the `UD-` stages (no recording of one, LS-A) left out.
LMS_QUANTS: tuple[str, ...] = tuple(
    stage for stage in QUANT_ORDER if stage not in _TENSOR_STAGES and not stage.startswith("UD-")
)

NO_SAFE_COMMAND = "no safe command"
# `lms get <owner>/<name>` names an artifact of the LM Studio Hub; a Hugging Face repository is named by its URL.
HF_URL_PREFIX = "https://huggingface.co/"
COMPUTED_ONLY_PREFIX = "computed only: lms get for"
COMPUTED_ONLY_SPLIT = f"{COMPUTED_ONLY_PREFIX} split files is not checked"
COMPUTED_ONLY_SUBFOLDER = f"{COMPUTED_ONLY_PREFIX} files in a subfolder is not checked"
_REPO_ASCII_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
_WEIGHT_ROLES = ("weights", "weights_shard")


def is_lms_lock(reason: str | None) -> bool:
    """Whether `reason` is one of the lock texts of an LM Studio row (`lms_reason`)."""
    return reason is not None and (reason == NO_SAFE_COMMAND or reason.startswith(COMPUTED_ONLY_PREFIX))


def weight_files(package: Package) -> list:
    """The weight files of a package (`mmproj` and other extras are none)."""
    return [file for file in package.files if file.role in _WEIGHT_ROLES]


def _safe_repo(repo: object) -> bool:
    return isinstance(repo, str) and _HF_REPO_RE.fullmatch(repo) is not None and _REPO_ASCII_RE.fullmatch(repo) is not None


def lms_reason(package: Package) -> str | None:
    """Why `package` has no `lms get` line, or `None` where it has one.

    `no safe command` for a name or stage that is not safe to put in a command; `computed only: ...` for a
    package whose download `lms get` is not checked for (split files, a file in a subfolder, a stage that is
    not in `LMS_QUANTS`).
    """
    stage = package.quantization
    if package.source != "huggingface" or package.format != "gguf" or not _safe_repo(package.repo) or stage not in QUANT_ORDER:
        return NO_SAFE_COMMAND
    files = weight_files(package)
    if not files:
        return NO_SAFE_COMMAND
    if len(files) > 1:
        return COMPUTED_ONLY_SPLIT
    name = files[0].name
    if "/" in name:
        return COMPUTED_ONLY_SUBFOLDER
    if not SAFE_REPO_PATH_RE.fullmatch(name):
        return NO_SAFE_COMMAND
    if stage not in LMS_QUANTS:
        return f"{COMPUTED_ONLY_PREFIX} {stage} is not checked"
    return None


def install_name(package: Package) -> str | None:
    """`LMSTUDIO.install_name`: `<owner>/<repo>@<quant>` for a package that has a command, else `None`."""
    return None if lms_reason(package) is not None else f"{package.repo}@{package.quantization}"


def command(name: str) -> str:
    """`LMSTUDIO.command`: the line in the `PATH` form,
    `lms get --gguf https://huggingface.co/<owner>/<repo>@<quant>`.

    The URL, not `<owner>/<repo>`: `lms get <owner>/<name>` resolves an artifact of the LM Studio Hub, and
    `lms get --help` says a model from Hugging Face is downloaded by its full URL.
    """
    return f"lms get --gguf {HF_URL_PREFIX}{name}"


def home_command(name: str, system: str) -> str:
    """The call where `lms` is not in the `PATH` but where LM Studio put it: a fixed shell expression, never
    a resolved path (PowerShell under Windows, a POSIX shell elsewhere). `cmd.exe` does not run it."""
    if system == "win32":
        return f'& "$env:USERPROFILE\\.lmstudio\\bin\\lms.exe" get --gguf {HF_URL_PREFIX}{name}'
    return f'"$HOME/.lmstudio/bin/lms" get --gguf {HF_URL_PREFIX}{name}'


def lms_form(room: RoomProbes) -> str | None:
    """How the program can be named on this machine (`lmstudio_server.lms_call`: `path`, `home` or `None`).
    A path search and a file test; the program is never started."""
    return lms_call(system=room.system, which=room.which, home=room.home, is_file=room.exists)


def command_for(form: str | None, name: str, system: str) -> str:
    """The line to copy for a call form: the home call when `lms` is only in the home folder, else the `PATH` form."""
    return home_command(name, system) if form == "home" else command(name)


def on_disk_text(rank: int) -> str:
    """The line that replaces the command when the package is on disk already (S8)."""
    return f"#{rank} is on disk already (size checked)"


def hint_lines(form: str | None, system: str, context: int, with_command: bool = True) -> list[str]:
    """The lines under the command of a chosen LM Studio row: about the call form, then what to do in LM Studio.

    The last one is the negative branch of the measuring switch: LM Studio is found, not measured. Without
    a command (the package is on disk already) the lines about the call form go.
    """
    lines: list[str] = []
    if with_command and form == "home":
        shell = " (PowerShell)" if system == "win32" else ""
        lines.append(f"lms is not on the PATH yet; this line calls it where LM Studio put it{shell}")
    elif with_command and form is None:
        lines.append("if lms is not found: it is in .lmstudio/bin in your home folder")
    lines.append(
        f"in LM Studio: load it with context length {context}, then start its server "
        "(the server switch in LM Studio, or lms server start)"
    )
    lines.append("then run modelroom again: it finds the server; LM Studio is not measured yet")
    return lines


def hint_rows(form: str | None, system: str, context: int, with_command: bool = True) -> list[str]:
    """`hint_lines` broken between words to what a labeled row has left of the 100 characters (`JOIN_WIDTH` less the
    label's column), the words as they are; a command is never part of it, so a command is never cut."""
    from .intro import LABEL_COLUMN  # call time: the head imports of this module stay the small ones, a test holds it
    from .screen import JOIN_WIDTH

    width = JOIN_WIDTH - LABEL_COLUMN
    return [row for text in hint_lines(form, system, context, with_command) for row in textwrap.wrap(text, width, break_long_words=False)]


# --- the model folder and "already there" ---------------------------------------------------------------


def model_place(environ: Mapping[str, str], home: Path, system: str) -> list[Path]:
    """`LMSTUDIO.model_place`: the standard folder, `~/.lmstudio/models`. No variable and no settings file is read."""
    return [home / ".lmstudio" / "models"]


def _inside(base: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(base), os.path.normcase(path)]) == os.path.normcase(base)
    except ValueError:  # another drive, or one path absolute and the other not
        return False


def package_on_disk(package: Package, place: Path, probes: CacheProbes | None = None) -> bool | None:
    """Whether `package` lies in the folder `place` with the size of the catalog revision: `True` it does,
    `False` it does not (no folder, no file, another size), `None` where that cannot be told.

    Only a package that has a command is looked at; for a split package, a file in a subfolder, an unsafe
    name or a stage outside `LMS_QUANTS` the answer is `None` without a single file access. A link that
    leads the repo folder or the file out of `<place>/<owner>/<repo>/` is `None` too, and so is a missing
    file of a declared size of 0.
    """
    if lms_reason(package) is not None:
        return None
    probes = probes if probes is not None else CacheProbes()
    owner, repo = package.repo.split("/", 1)
    file = weight_files(package)[0]
    base = os.path.join(str(place), owner, repo)
    real_place = probes.realpath(str(place))
    real_base = probes.realpath(base)
    if os.path.normcase(real_base) != os.path.normcase(os.path.join(real_place, owner, repo)):
        return None  # the repo folder is a link that leads elsewhere: nothing below it is touched
    real = probes.realpath(os.path.join(base, file.name))
    if not _inside(real_base, real):
        return None
    try:
        info = probes.stat(real)
        found: int | None = info.st_size if stat.S_ISREG(info.st_mode) else None
    except (FileNotFoundError, NotADirectoryError):
        found = None
    except OSError:
        return None
    if found == file.size_bytes:
        return True
    return None if file.size_bytes == 0 else False


def on_disk(target: object, packages: Sequence[Package]) -> frozenset:
    """`LMSTUDIO.on_disk`: the identities of those `packages` that lie in LM Studio's folder on the machine of
    the run (size checked); empty where the run has no room to read."""
    room = getattr(target, "room", None)
    if room is None:
        return frozenset()
    place = model_place(room.environ, room.home, room.system)[0]
    return frozenset(package_identity_key(package) for package in packages if package_on_disk(package, place) is True)


def lmstudio_disk_at(room: RoomProbes, packages: Sequence[Package] = ()) -> Disk:
    """The free disk where LM Studio keeps its models, read now (the nearest folder that exists), with
    `Disk.held` the packages already there: they need no disk (`pull_need_bytes`). Nothing of `Disk.needs`:
    a package that is not there needs its whole download."""
    place = model_place(room.environ, room.home, room.system)[0]
    base = free_disk_at(room, place)
    held = frozenset(package_identity_key(package) for package in packages if package_on_disk(package, place) is True)
    return replace(base, held=held)


def lmstudio_disk(disk: Disk, room: RoomProbes, config: "Configuration", here: str, packages: Sequence[Package]) -> Disk:
    """`disk` with what a machine that names LM Studio needs: its runtimes (`Disk.runtimes`, so the Markdown
    file judges a row as the terminal does) and LM Studio's own place (`Disk.lmstudio`, F6 a).

    The profile of `here` is read the way the render reads it (`[machines.<here>].profile` out of the
    profiles of the folder); a machine without LM Studio, without an entry, without a profile file, gets
    the very same `disk` back.
    """
    from .importer import scan_profiles  # call time: keeps this module's imports as small as hf_cache's
    from .runtimes import machine_runtime_keys  # call time: runtimes.py imports this module (cycle)

    machine = config.machines.get(here)
    profile = scan_profiles(config.paths.hardware_dir).profiles.get(machine.profile) if machine is not None and machine.profile else None
    keys = machine_runtime_keys(profile)
    if "lmstudio" not in keys:
        return disk
    return replace(disk, runtimes=keys, lmstudio=lmstudio_disk_at(room, packages))


__all__ = [
    "COMPUTED_ONLY_PREFIX",
    "COMPUTED_ONLY_SPLIT",
    "COMPUTED_ONLY_SUBFOLDER",
    "HF_URL_PREFIX",
    "LMS_QUANTS",
    "NO_SAFE_COMMAND",
    "command",
    "command_for",
    "hint_lines",
    "hint_rows",
    "home_command",
    "install_name",
    "is_lms_lock",
    "lms_form",
    "lms_reason",
    "lmstudio_disk",
    "lmstudio_disk_at",
    "model_place",
    "on_disk",
    "on_disk_text",
    "package_on_disk",
    "weight_files",
]
