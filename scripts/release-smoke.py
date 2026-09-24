"""Delivery smoke test: does the built wheel work on its own, outside this repository?

    uv run --frozen python scripts/release-smoke.py

Steps, each reported with its exit code and output: build the wheel; create a fresh venv in
the system temp directory (outside any git repository) and `pip install` the wheel there (not
editable, not from the tree; pip fetches `pydantic` from the package index); write a customer
configuration derived from `modelroom.example.toml` (one machine `kunde`, reserves 0, no rating
source); `modelroom hardware`, which must write exactly one schema-2 profile; place the
schema-1 profile the renderer still reads, built from the recorded fixtures; place a snapshot
built from the recorded fixtures in `tests/`
(no live data); `modelroom fetch` with the network blocked through an unreachable proxy (the
contract honours `HTTPS_PROXY`), which must end every area incomplete with exit 1, keep the
packages and leave the lock free; `modelroom render`; delete the temp directory.

The report goes to stdout. Exit code: 0 every step as expected, 1 a step did not behave as
expected, 2 the environment could not be set up (build, venv or install failed).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
MACHINE = "kunde"
BLOCKING_PROXY = "http://127.0.0.1:9"
FIXTURE_RUN_AT = datetime(2026, 9, 22, 9, tzinfo=timezone.utc)
LOCK_PROBE = (
    "import sys; from datetime import datetime, timezone; from pathlib import Path; "
    "from modelroom.state import acquire_lock, release_lock; "
    "release_lock(acquire_lock(Path(sys.argv[1]), 'smoke', datetime.now(timezone.utc)))"
)


class Step(NamedTuple):
    name: str
    exit_code: int | None
    severity: int  # 0 as expected, 1 not as expected, 2 environment
    detail: str


class SetupError(Exception):
    """The environment for the smoke test could not be set up."""


# -- pieces with logic of their own (unit tested) -----------------------------------------------


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(value)


def _table(header: str, values: dict) -> list[str]:
    return [header] + [f"{key} = {_toml_value(value)}" for key, value in values.items()] + [""]


def customer_config(example_text: str, base_models: set[str]) -> str:
    """The example configuration cut to the families the fixture snapshot covers, one machine."""
    example = tomllib.loads(example_text)
    missing = [key for key in ("schema_version", "packagers", "publishers", "families", "paths", "llmfit")
               if key not in example]
    if missing:
        raise ValueError(f"the example configuration lacks {', '.join(missing)}")
    families = [family for family in example["families"]
                if all(model["hf_repo"] in base_models for model in family["base_models"])]
    if not families:
        raise ValueError(f"no family of the example configuration covers {sorted(base_models)}")
    lines = [f"schema_version = {example['schema_version']}",
             f"packagers = {_toml_value(example['packagers'])}",
             f"publishers = {_toml_value(example['publishers'])}", ""]
    for family in families:
        lines += _table("[[families]]", {"name": family["name"]})
        for model in family["base_models"]:
            lines += _table("  [[families.base_models]]", model)
    lines += _table(f"[machines.{MACHINE}]", {"reserve_ram_gib": 0, "reserve_vram_gib": 0, "writer": True})
    lines += _table("[paths]", example["paths"])
    lines += _table("[llmfit]", example["llmfit"])
    return "\n".join(lines)


def markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    """The rows of the first table under `## <heading>`, keyed by the header cells (`\\|` stays in its cell)."""
    lines = text.splitlines()
    if f"## {heading}" not in lines:
        return []
    table = []
    for line in lines[lines.index(f"## {heading}") + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            table.append([cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip()[1:-1])])
    if len(table) < 2:
        return []
    header, rows = table[0], table[2:]
    return [dict(zip(header, row)) for row in rows]


def render_problems(text: str, machine: str, installed: int | None) -> list[str]:
    """What a customer's document must show: no stars, a fit column, the profile's own inventory.

    `installed` is the number of models the machine's hardware profile recorded, `None` when the
    profile recorded the inventory as unavailable.
    """
    problems = []
    packages = markdown_table(text, "Packages")
    column = f"fit (computed, v1): {machine}"
    if not packages:
        problems.append("Packages table is empty")
    if any(row.get("Stars") != "–" for row in packages):
        problems.append("a Stars cell is not '–' although no rating source is configured")
    if any(not row.get(column) for row in packages):
        problems.append(f"column '{column}' missing or empty")
    machines = {row.get("Machine"): row for row in markdown_table(text, "Machines")}
    cell = machines.get(machine, {}).get("Installed", "")
    if (cell != str(installed)) if installed is not None else not re.fullmatch(r"unknown \(.+\)", cell):
        problems.append(f"Machines table: Installed for {machine} is {cell!r}, profile says {installed}")
    return problems


def expected_areas(config: dict) -> set[tuple[str, str, str | None]]:
    """The areas `fetch` owes for a configuration (CONTRACTS.md, "Area semantics"): per base model one
    Hugging Face area per packager and for the base model's own owner, if allowed, plus Ollama."""
    allowed = set(config["packagers"]) | set(config["publishers"])
    areas = set()
    for family in config["families"]:
        for model in family["base_models"]:
            repo = model["hf_repo"]
            owners = dict.fromkeys([*config["packagers"], repo.split("/", 1)[0]])
            areas |= {("huggingface", repo, owner) for owner in owners if owner in allowed}
            if model.get("ollama_base") and model.get("ollama_tag"):
                areas.add(("ollama", repo, None))
    return areas


def area_problems(areas: list[dict], expected: set[tuple[str, str, str | None]]) -> list[str]:
    seen = [(area["source"], area["base_model_hf_repo"], area["packager"]) for area in areas]
    problems = []
    if len(seen) != len(set(seen)) or set(seen) != expected:
        problems.append(f"areas {sorted(map(str, seen))} are not exactly the expected {sorted(map(str, expected))}")
    complete = [area for area in areas if area["status"] != "incomplete"]
    if complete:
        problems.append(f"{len(complete)} of {len(areas)} areas not incomplete without network")
    return problems


def format_report(steps: list[Step]) -> str:
    lines = []
    for number, step in enumerate(steps, start=1):
        verdict = "OK" if step.severity == 0 else "FAIL"
        lines.append(f"[{number}] {step.name}: exit {step.exit_code}, {verdict}")
        lines += [f"    {line}" for line in step.detail.splitlines() if line.strip()]
    return "\n".join(lines)


def worst(steps: list[Step]) -> int:
    return max((step.severity for step in steps), default=0)


# -- the run -----------------------------------------------------------------------------------


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Exit code and output; a program that is not there is exit 127, as a shell reports it."""
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                                errors="replace")
    except FileNotFoundError:
        return 127, f"not found: {command[0]}"
    return result.returncode, (result.stdout + result.stderr).strip()


def tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.splitlines()[-lines:])


def clean_env(**extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in {"PYTHONPATH", "VIRTUAL_ENV", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}}
    env.update(extra)
    return env


def outside_any_repository(path: Path) -> bool:
    return not any((candidate / ".git").exists() for candidate in [path, *path.parents])


def step_build(work: Path) -> tuple[Step, Path]:
    code, output = run(["uv", "build", "--wheel", "--out-dir", str(work / "dist"), str(REPO)], REPO)
    wheels = sorted((work / "dist").glob("*.whl"))
    if code != 0 or len(wheels) != 1:
        raise SetupError(f"uv build: exit {code}\n{tail(output)}")
    return Step("build wheel", code, 0, wheels[0].name), wheels[0]


def step_install(work: Path, wheel: Path) -> tuple[Step, Path]:
    if not outside_any_repository(work):
        raise SetupError(f"temp directory lies inside a git repository: {work}")
    venv = work / "venv"
    code, output = run([sys.executable, "-m", "venv", str(venv)], work, clean_env())
    if code != 0:
        raise SetupError(f"venv: exit {code}\n{tail(output)}")
    python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    code, output = run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", str(wheel)],
                       work, clean_env())
    if code != 0:
        raise SetupError(f"pip install: exit {code}\n{tail(output)}")
    _, location = run([str(python), "-c", "import modelroom; print(modelroom.__file__)"], work, clean_env())
    severity = 0 if Path(location).is_relative_to(venv) else 1
    return Step("fresh venv + pip install wheel", code, severity, f"{tail(output, 2)}\nmodelroom from: {location}"), python


def step_config(customer: Path) -> tuple[Step, Path]:
    fixture = fixture_snapshot(customer.parent / "fixture")
    base_models = {model["hf_repo"] for model in fixture["base_models"]}
    config = customer / "modelroom.toml"
    customer.mkdir()
    config.write_text(customer_config((REPO / "modelroom.example.toml").read_text(encoding="utf-8"), base_models),
                      encoding="utf-8")
    return Step("customer configuration", 0, 0, f"machine {MACHINE}, reserves 0, no rating source\n"
                f"families for {sorted(base_models)}"), config


def fixture_snapshot(folder: Path) -> dict:
    """A snapshot built by this repository's own code from the recorded fixtures in tests/."""
    sys.path.insert(0, str(REPO / "tests"))
    from fixture_support import build_transport, qwen35_example_config_dict, qwen35_transport_mapping
    from modelroom.cli import fetch_with_config
    from modelroom.config import Configuration

    config = Configuration.from_dict(qwen35_example_config_dict(str(folder / "state"), str(folder / "models.md")))
    code = fetch_with_config(config, "workstation", build_transport(qwen35_transport_mapping()), FIXTURE_RUN_AT)
    if code != 0:
        raise SetupError(f"fixture snapshot: fetch against the recorded fixtures ended with exit {code}")
    return json.loads(config.paths.snapshot_file.read_text(encoding="utf-8"))


def modelroom(python: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    executable = python.parent / "modelroom"
    return run([str(executable), *args], python.parent, env or clean_env())


def step_hardware(python: Path, config: Path) -> Step:
    """`modelroom hardware` on this machine: exactly one schema-2 profile must be written.

    The file is named after the random `profile_id`, so the check is "one new schema-2 profile
    in the folder", not a file name (CONTRACTS.md, "Hardware measurement").
    """
    # The pointer file lives in the user's home folder, so the run gets a home of its own inside
    # the temp tree: a smoke test must not leave a binding to a deleted folder in anybody's home.
    home = str(config.parent)
    code, output = modelroom(python, "hardware", "--config", str(config), "--machine", MACHINE,
                             env=clean_env(USERPROFILE=home, HOME=home))
    written = [path for path in sorted((config.parent / "state" / "hardware").glob("*.json"))
               if json.loads(path.read_text(encoding="utf-8")).get("schema_version") == 2]
    ok = code == 0 and len(written) == 1
    return Step("modelroom hardware", code, 0 if ok else 1,
                f"{tail(output)}\nschema-2 profiles written: {[path.name for path in written]}")


def step_place_v1_profile(config: Path) -> Step:
    """Place the schema-1 profile `render` still reads, built from this repository's fixtures.

    `hardware` writes schema 2; the renderer switches to it in its own work package and until
    then reads `<machine>.json`. Built from the recorded `llmfit system --json` and
    `/api/tags` fixtures, never from live data -- so the render step below keeps checking the
    renderer, not the measurement.
    """
    profile = v1_profile_from_fixtures(MACHINE)
    path = config.parent / "state" / "hardware" / f"{MACHINE}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8", newline="\n")
    return Step("place schema-1 profile for render", 0, 0,
                f"{len(profile['installed'])} installed models, vram {profile['vram_gib']} GiB")


def v1_profile_from_fixtures(machine: str) -> dict:
    """A valid schema-1 `HardwareSnapshot` dict for `machine`, from the recorded fixtures."""
    sys.path.insert(0, str(REPO / "tests"))
    from modelroom.contracts import load_hardware_snapshot

    fixtures = REPO / "tests" / "fixtures"
    system = json.loads((fixtures / "llmfit_system_laptop.json").read_text(encoding="utf-8"))["system"]
    tags = json.loads((fixtures / "ollama_tags_local.json").read_text(encoding="utf-8"))["models"]
    data = {
        "schema_version": 1,
        "machine": machine,
        "measured_at": FIXTURE_RUN_AT.isoformat(),
        "llmfit_version": "0.0.0",
        "vram_gib": system["gpu_vram_gb"],
        "ram_gib": system["total_ram_gb"],
        "free_ram_gib_at_measurement": system["available_ram_gb"],
        "gpu_name": system["gpu_name"],
        "backend": system["backend"],
        "unified_memory": system["unified_memory"],
        "installed": [
            {
                "name": entry["name"],
                "digest": f"sha256:{entry['digest']}",
                "size_bytes": entry.get("size", 0),
                "observed_at": FIXTURE_RUN_AT.isoformat(),
            }
            for entry in tags
        ],
        "installed_unavailable_reason": None,
        "measurements": [],
    }
    return load_hardware_snapshot(data).model_dump(mode="json")


def step_place_snapshot(config: Path) -> Step:
    source = config.parent.parent / "fixture" / "state" / "modelroom.json"
    target = config.parent / "state" / "modelroom.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    snapshot = json.loads(target.read_text(encoding="utf-8"))
    return Step("place fixture snapshot", 0, 0, f"{len(snapshot['packages'])} packages, run_at {snapshot['run_at']}")


def step_fetch_offline(python: Path, config: Path) -> Step:
    snapshot_file = config.parent / "state" / "modelroom.json"
    before = json.loads(snapshot_file.read_text(encoding="utf-8"))["packages"]
    proxy = {"HTTPS_PROXY": BLOCKING_PROXY, "HTTP_PROXY": BLOCKING_PROXY}
    code, output = modelroom(python, "fetch", "--config", str(config), "--machine", MACHINE, env=clean_env(**proxy))
    after = json.loads(snapshot_file.read_text(encoding="utf-8"))
    area_issues = area_problems(after["areas"], expected_areas(tomllib.loads(config.read_text(encoding="utf-8"))))
    problems = ([] if code == 1 else [f"expected exit 1, got {code}"]) + area_issues
    if after["packages"] != before:
        problems.append("packages changed although no area completed")
    lock_code, lock_output = run([str(python), "-c", LOCK_PROBE, str(config.parent / "state" / "modelroom.lock")],
                                 config.parent, clean_env())
    if lock_code != 0:
        problems.append(f"lock not free after fetch: {tail(lock_output, 3)}")
    incomplete = sum(area["status"] == "incomplete" for area in after["areas"])
    summary = f"areas incomplete: {incomplete}/{len(after['areas'])}, exactly the expected areas: {not area_issues}, " \
              f"packages kept: {after['packages'] == before}, lock free: {lock_code == 0}"
    first_error = next((area["error"] for area in after["areas"] if area["error"]), None)
    return Step("modelroom fetch, network blocked", code, 1 if problems else 0,
                "\n".join(problems + [summary, f"first area error: {first_error}", tail(output, 6)]))


def profile_installed(config: Path) -> int | None:
    profile = json.loads((config.parent / "state" / "hardware" / f"{MACHINE}.json").read_text(encoding="utf-8"))
    return None if profile["installed"] is None else len(profile["installed"])


def step_render(python: Path, config: Path) -> Step:
    code, output = modelroom(python, "render", "--config", str(config))
    document = config.parent / "docs" / "models.md"
    problems = [] if code == 0 else [f"expected exit 0, got {code}"]
    detail = []
    if document.is_file():
        text = document.read_text(encoding="utf-8")
        installed = profile_installed(config)
        problems += render_problems(text, MACHINE, installed)
        detail = [f"{row.get('Packager')}: stars {row.get('Stars')}, {row.get(f'fit (computed, v1): {MACHINE}')}"
                  for row in markdown_table(text, "Packages")]
        detail.append(f"installed ({MACHINE}) rendered from the profile's {installed} observed models")
    else:
        problems.append("no document written")
    return Step("modelroom render", code, 1 if problems else 0, "\n".join(problems + detail + [tail(output, 3)]))


def guarded(name: str, step: Callable[[], Step]) -> Step:
    """A step that raises (a missing file, an unexpected shape) is a failed step, not a crash."""
    try:
        return step()
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return Step(name, None, 1, f"{type(exc).__name__}: {exc}")


def run_steps(work: Path) -> list[Step]:
    steps: list[Step] = []
    try:
        step, wheel = step_build(work)
        steps.append(step)
        step, python = step_install(work, wheel)
        steps.append(step)
        step, config = step_config(work / "kunde")
        steps.append(step)
    except (SetupError, OSError, KeyError, ValueError, TypeError) as exc:
        return steps + [Step("setup", None, 2, f"{type(exc).__name__}: {exc}")]
    steps.append(guarded("modelroom hardware", lambda: step_hardware(python, config)))
    steps.append(guarded("place schema-1 profile for render", lambda: step_place_v1_profile(config)))
    steps.append(guarded("place fixture snapshot", lambda: step_place_snapshot(config)))
    steps.append(guarded("modelroom fetch, network blocked", lambda: step_fetch_offline(python, config)))
    steps.append(guarded("modelroom render", lambda: step_render(python, config)))
    return steps


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        work = Path(tempfile.mkdtemp(prefix="modelroom-smoke-"))
    except OSError as exc:
        steps = [Step("setup", None, 2, f"temp directory: {type(exc).__name__}: {exc}")]
    else:
        try:
            steps = run_steps(work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        steps.append(Step("delete temp directory", None, 1 if work.exists() else 0, f"removed: {not work.exists()}"))
    print(format_report(steps))
    print(f"RELEASE-SMOKE: {'OK' if worst(steps) == 0 else 'FAIL'}")
    return worst(steps)


if __name__ == "__main__":
    sys.exit(main())
