"""AP9-LS, AP9-LS-G: no module of LM Studio starts a process, with one exception: `lmstudio_get.py`.

Three guards, each with a test of its own so none can pass for the wrong reason: a static check of the
source of the LM Studio modules (`ast`), a trap on `subprocess.Popen` and `os.system` that fails the test
at the first call through either function, and a run of everything LM Studio does with a program called `lms`
first in the `PATH` and in the home folder. The run proves it by the trap and by a mark file the
program would leave.

The exception (AP9-LS-G, the download after a yes): `modelroom/lmstudio_get.py` imports `subprocess` in the
ordinary way and has exactly one call, `subprocess.run(argv, check=False)` inside `run_lms`: one positional
argument, the name `argv`, and one keyword argument, `check=False`. `shell=`, `*args`, `**kwargs`, a second
call and a call outside `run_lms` are findings of `get_findings`; every other process path stays forbidden
there as in the other modules. The whole guided run, with the yes and the no, is `test_lmstudio_runs_lms_once.py`.
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

import pytest

from modelroom.guided_lmstudio import ROW, card_facts, measure, offer, start_fact
from modelroom.lmstudio_server import MODELS_PATH, detect_lmstudio, lms_call, read_lmstudio_models
from modelroom.measurements import Scenario
from modelroom.runtimes import Origins

from fixture_lmstudio import chat_calls, fake_lms_on_path, lms_in_the_home_folder, lmstudio_servers, run_of, trap_processes

PACKAGE = Path(__file__).resolve().parent.parent / "modelroom"
LMSTUDIO_MODULES = ("lmstudio_server", "lmstudio_install", "loadtest_lmstudio", "guided_lmstudio")
_PROCESS_MODULES = ("os", "asyncio", "subprocess")
_OS_PROCESS = ("system", "popen")
_OS_PREFIXES = ("exec", "spawn")


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> module for `import os`, `import os as operating`, `import asyncio as aio`."""
    names = {"os": "os", "asyncio": "asyncio"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("os", "asyncio"):
                    names[alias.asname or alias.name] = alias.name
    return names


def process_findings(source: str) -> list[str]:
    """What in `source` can start a process: an import of `subprocess`, a call of `os.system`, `os.popen`,
    `os.exec*`, `os.spawn*` or `asyncio.create_subprocess_*`. **Scope, stated as a limit:** the check knows
    the ordinary import, the alias, the star import and the dynamic import with a constant module name
    (`__import__`, `importlib.import_module`, positional or `name=`). The trap on `subprocess.Popen` and
    `os.system` in `test_modelroom_never_runs_lms` catches the calls that go through those two; a call of
    `os.exec*` or `os.spawn*` is guarded statically, within the import and call forms listed here, and a
    form outside them (`getattr`, a name built at run time) is not covered by this test."""
    findings = []
    tree = ast.parse(source)
    names = _aliases(tree)
    loaders = _loaders(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            findings += [f"import {alias.name}" for alias in node.names if alias.name.split(".")[0] == "subprocess"]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] == "subprocess":
                findings.append(f"from {module} import ...")
            for alias in node.names:
                if module in ("os", "asyncio") and alias.name == "*":
                    findings.append(f"from {module} import *")
                if module == "os" and _is_os_process(alias.name):
                    findings.append(f"from os import {alias.name}")
                if module == "asyncio" and alias.name.startswith("create_subprocess"):
                    findings.append(f"from asyncio import {alias.name}")
        elif isinstance(node, ast.Call) and _dynamic_import(node, loaders) in _PROCESS_MODULES:
            findings.append(f"dynamic import of {_dynamic_import(node, loaders)}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = names.get(node.func.value.id) if isinstance(node.func.value, ast.Name) else None
            if owner == "os" and _is_os_process(node.func.attr):
                findings.append(f"os.{node.func.attr}(...)")
            if owner == "asyncio" and node.func.attr.startswith("create_subprocess"):
                findings.append(f"asyncio.{node.func.attr}(...)")
    return findings


def _loaders(tree: ast.AST) -> set[str]:
    """The local names of `importlib.import_module`: `from importlib import import_module [as load]`, or
    `from importlib import *`."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    names.add(alias.asname or alias.name)
                elif alias.name == "*":
                    names.add("import_module")
    return names


def _dynamic_import(call: ast.Call, loaders: set[str]) -> str | None:
    """The first part of the module a call of `__import__` or `importlib.import_module` (also under a local
    name) names by a constant string, positional or as `name=`."""
    func = call.func
    named = (
        (isinstance(func, ast.Name) and (func.id == "__import__" or func.id in loaders))
        or (isinstance(func, ast.Attribute) and func.attr == "import_module")
    )
    argument = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "name"), None)
    if named and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value.split(".")[0]
    return None


def _is_os_process(name: str) -> bool:
    return name in _OS_PROCESS or name.startswith(_OS_PREFIXES)


# --- the static check ---------------------------------------------------------------------------------


def test_no_lmstudio_module_starts_a_process():
    """Every module of LM Studio that exists is read; one that is not there yet is no fault, so the check
    runs from the first of them on (the later ones are added by later sub-packages). `guided_install.py`,
    where the question is asked, starts no process either: the one call lives in `lmstudio_get.py`."""
    present = [name for name in (*LMSTUDIO_MODULES, "guided_install") if (PACKAGE / f"{name}.py").is_file()]
    assert "lmstudio_server" in present and "guided_install" in present
    for name in present:
        assert process_findings((PACKAGE / f"{name}.py").read_text(encoding="utf-8")) == [], name


# --- the one exception: lmstudio_get.py ---------------------------------------------------------------


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def get_findings(source: str) -> list[str]:
    """What in the source of `lmstudio_get.py` is more than the one allowed process call.

    Allowed: the ordinary `import subprocess` (no alias) and exactly one call `subprocess.run(argv, check=False)`
    inside the function `run_lms`: one positional argument, the name `argv`, one keyword argument, `check=False`.
    Everything `process_findings` knows stays a finding (the import of `subprocess` aside), and so is any other
    use of the name `subprocess`: a second call, `shell=`, `*args`, `**kwargs`, a missing or true `check`, a
    call outside `run_lms`, the module passed on or its function taken as a value."""
    tree = ast.parse(source)
    findings = [finding for finding in process_findings(source) if finding != "import subprocess"]
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import) and any(a.name == "subprocess" for a in node.names)]
    if len(imports) != 1 or any(alias.asname is not None for alias in imports[0].names if alias.name == "subprocess"):
        findings.append("the ordinary `import subprocess` is missing, repeated or aliased")
    run_lms = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_lms"]
    inside = {id(node) for function in run_lms for node in ast.walk(function)}
    uses = [node for node in ast.walk(tree) if _is_name(node, "subprocess")]
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _is_name(node.func.value, "subprocess")
    ]
    if len(calls) != 1 or len(uses) != 1:
        findings.append(f"{len(calls)} call(s) through subprocess and {len(uses)} use(s) of its name, not exactly one")
    for call in calls:
        keywords = call.keywords
        exact = (
            call.func.attr == "run" and len(call.args) == 1 and _is_name(call.args[0], "argv") and len(keywords) == 1
            and keywords[0].arg == "check" and isinstance(keywords[0].value, ast.Constant) and keywords[0].value.value is False
        )
        if not exact:
            findings.append("the call is not `subprocess.run(argv, check=False)`")
        if id(call) not in inside:
            findings.append("the call is outside `run_lms`")
    return findings


GET_POSITIVE = '''
import subprocess


def run_lms(argv):
    return subprocess.run(argv, check=False).returncode
'''

GET_NEGATIVE = {
    "a second run call": GET_POSITIVE + "\ndef again(argv):\n    return subprocess.run(argv, check=False)\n",
    "shell=True": GET_POSITIVE.replace("check=False", "check=False, shell=True"),
    "**kwargs": GET_POSITIVE.replace("argv, check=False", "argv, **options"),
    "*args": GET_POSITIVE.replace("argv, check=False", "*argv, check=False"),
    "no check": GET_POSITIVE.replace("argv, check=False", "argv"),
    "check=True": GET_POSITIVE.replace("check=False", "check=True"),
    "a second positional argument": GET_POSITIVE.replace("argv, check", "argv, 3, check"),
    "another name than argv": GET_POSITIVE.replace("subprocess.run(argv", "subprocess.run(command"),
    "a call outside run_lms": GET_POSITIVE.replace("def run_lms", "def not_run_lms"),
    "Popen in place of run": GET_POSITIVE.replace("subprocess.run", "subprocess.Popen"),
    "the function taken as a value": GET_POSITIVE + "\nstart = subprocess.run\n",
    "an aliased import": GET_POSITIVE.replace("import subprocess", "import subprocess as sp"),
    "os.system beside it": GET_POSITIVE + "\nimport os\nos.system('x')\n",
    "a dynamic import": GET_POSITIVE + "\n__import__('subprocess')\n",
    "no import": GET_POSITIVE.replace("import subprocess\n", "import json\n"),
}


def test_the_one_exception_is_the_module_that_runs_lms_get():
    """`lmstudio_get.py` exists, and holds nothing but the one `subprocess.run(argv, check=False)` in `run_lms`."""
    path = PACKAGE / "lmstudio_get.py"
    assert path.is_file(), "modelroom/lmstudio_get.py does not exist: the one place that runs lms get is missing"
    source = path.read_text(encoding="utf-8")
    assert get_findings(source) == []
    assert source.count("subprocess.run(") == 1 and "def run_lms(" in source


def test_the_exception_form_passes_and_every_other_form_is_a_finding():
    assert get_findings(GET_POSITIVE) == []
    for name, source in GET_NEGATIVE.items():
        assert get_findings(source) != [], name


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess",
        "import subprocess.run as r",
        "from subprocess import run",
        "import os\nos.system('x')",
        "import os\nos.popen('x')",
        "import os\nos.execv('x', [])",
        "import os\nos.spawnl(0, 'x')",
        "from os import system",
        "import asyncio\nasyncio.create_subprocess_exec('x')",
        "from asyncio import create_subprocess_shell",
        "import os as operating\noperating.spawnv(0, 'x', [])",
        "import os as operating\noperating.system('x')",
        "import asyncio as aio\naio.create_subprocess_exec('x')",
        "from os import *\nspawnv(0, 'x', [])",
        "from subprocess import *",
        "from asyncio import *",
        "__import__('os').execv('x', [])",
        "__import__('subprocess')",
        "__import__('os.path')\n__import__('asyncio')",
        "import importlib\nimportlib.import_module('subprocess')",
        "from importlib import import_module as load\nload('os').execv('x', [])",
        "from importlib import import_module\nimport_module('subprocess')",
        "import importlib\nimportlib.import_module(name='os')",
        "from importlib import import_module as load\nload(name='asyncio')",
        "from importlib import *\nimport_module('os').execv('x', [])",
    ],
)
def test_the_static_check_finds_what_it_looks_for(source):
    assert process_findings(source) != []


def test_the_static_check_leaves_an_ordinary_module_alone():
    assert process_findings("import os\nimport shutil\nshutil.which('x')\nos.path.join('a', 'b')") == []
    assert process_findings("import os as operating\noperating.path.join('a', 'b')") == []
    assert process_findings("from importlib import *\nimport_module('json')") == []
    assert process_findings("from importlib import import_module as load\nload(name='json')\nload('json')") == []
    assert process_findings("__import__('json')\nfrom json import *\nimport importlib\nimportlib.import_module('json')") == []


# --- the trap -----------------------------------------------------------------------------------------


def test_the_trap_catches_a_direct_and_a_shell_call(monkeypatch):
    """A direct call, a call by an absolute path and a shell call each fail, whatever the first argument is."""
    import os
    import subprocess

    trap_processes(monkeypatch)

    for call in (
        lambda: subprocess.Popen(["lms", "ls"]),
        lambda: subprocess.Popen(["/absolute/path/to/helper", "ls"]),
        lambda: subprocess.Popen("lms ls --json", shell=True),
        lambda: subprocess.run(["anything"]),
        lambda: os.system("lms ls"),
        lambda: os.popen("lms ls"),
    ):
        with pytest.raises(pytest.fail.Exception, match="a process was started"):
            call()


# --- the run ------------------------------------------------------------------------------------------


def test_modelroom_never_runs_lms_from_the_home_folder_either(tmp_path, monkeypatch):
    """No program in the `PATH`, only `~/.lmstudio/bin/lms` (and `lms.exe`) in the test home, built so that a
    run would leave a mark: the same functions under the same trap, and no mark."""
    mark = tmp_path / "lms-was-run.txt"
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    home = tmp_path / "home"
    lms_in_the_home_folder(home, mark)
    assert shutil.which("lms") is None
    trap_processes(monkeypatch)
    run = run_of(lmstudio_servers(["lmstudio_v1_models_loaded.json"]))
    scenario = Scenario(context_requested=8192, context_origin="default", kv_type="f16", kv_type_assumed=True, requests=1)

    assert detect_lmstudio(None, system=sys.platform, which=shutil.which, home=home) == "installed"
    assert lms_call(system=sys.platform, which=shutil.which, home=home) == "home"
    fact = start_fact(Origins(daemon=None, servers=run.servers))
    given = offer(run, None, scenario, tmp_path)
    said = measure(run, None, scenario, tmp_path, given, [ROW])

    assert fact is not None and given.rows == () and said is None
    assert chat_calls(run.servers.transports["lmstudio"]) == []
    assert not mark.exists()


def test_modelroom_never_runs_lms(tmp_path, monkeypatch):
    """A program called `lms` stands first in the `PATH` and in the home folder, the server lists a loaded
    model: detection, the call form, the reader and all four functions of the entry (start row, offer,
    measurement, card) run, and no process starts (the trap) and the program leaves no mark. The whole
    guided run with a registered entry is the same test in AP9-LS-E."""
    mark = fake_lms_on_path(tmp_path, monkeypatch)
    home = tmp_path / "home"
    lms_in_the_home_folder(home)
    assert shutil.which("lms") is not None
    trap_processes(monkeypatch)
    server = lmstudio_servers(["lmstudio_v1_models_loaded.json"])

    state = detect_lmstudio(server, system=sys.platform, which=shutil.which, home=home)
    installed = detect_lmstudio(None, system=sys.platform, which=shutil.which, home=home)
    form = lms_call(system=sys.platform, which=shutil.which, home=home)
    models = read_lmstudio_models(server)

    assert (state, installed, form) == ("serving", "installed", "path")
    assert [loaded.instance_id for loaded in models.loaded] == ["qwen3.5-4b"]
    assert [call[:2] for call in server.calls] == [("GET", MODELS_PATH)] * 2

    run = run_of(lmstudio_servers(["lmstudio_v1_models_loaded.json"]))
    scenario = Scenario(context_requested=8192, context_origin="default", kv_type="f16", kv_type_assumed=True, requests=1)
    fact = start_fact(Origins(daemon=None, servers=run.servers))
    given = offer(run, None, scenario, tmp_path)
    said = measure(run, None, scenario, tmp_path, given, [ROW])

    assert fact is not None and given.rows == () and said is None and card_facts(run) == ()
    assert chat_calls(run.servers.transports["lmstudio"]) == []
    assert not mark.exists()
