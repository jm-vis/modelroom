"""The sink of machine names (AP9-FR, FR-N): `modelroom/neutral.py` and `modelroom/neutral_sweep.py`.

No file of modelroom carries a machine name: not this computer's, not one from an export, a record,
a backup or an answer of the daemon. The sentinels (`tests/fixture_neutral.py`) stand for such names;
every test looks for them in the bytes of every file a run leaves behind.
"""

from __future__ import annotations

import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fixture_neutral import SENTINEL_HOST, SENTINEL_MODEL, SENTINEL_VERSIONS, THIS_MACHINE, files_with
from fixture_support import (
    DEEPSEEK_OLLAMA_NAME,
    DEEPSEEK_REPO,
    DEEPSEEK_SHA,
    DEEPSEEK_WEIGHTS_DIGEST,
    GRANITE_MANIFEST_DIGEST,
    PROFILE_ID,
    build_transport,
    default_generates,
    granite_ollama_package,
    guided_transport_mapping,
    offline_daemon,
    offline_servers,
    v2_profile,
    windows_probes,
)
from modelroom import neutral, remote_store
from modelroom.binding import read_pointer
from modelroom.cli import main
from modelroom.contracts import BaseModelSpec, Snapshot
from modelroom.daemon import GENERATE_PATH, PS_PATH, TAGS_PATH, VERSION_PATH, FixtureDaemon
from modelroom.dialog import FileAsker
from modelroom.guided import run_guided
from modelroom.http import Response
from modelroom.importer import import_profile
from modelroom.loadtest import installed_candidates, installed_models, run_load_test, store_measurement
from modelroom.measurements import MeasurementRecord, default_scenario
from modelroom.neutral import (
    ANOTHER_MODEL,
    DIGEST_CHANGED,
    FAILED,
    NO_OBSERVATION,
    UNKNOWN_COMPARABLE,
    UNKNOWN_VALIDITY,
    Names,
    comparable_reason,
    display_name_for,
    machine_keys,
    neutral_record,
    neutral_version,
    record_name,
    validity_reason,
)
from modelroom.neutral_sweep import sweep
import modelroom.state as state_module
from modelroom.state import acquire_lock, release_lock

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures"
EXPORT = json.loads((FIXTURES / "export_v1.json").read_text(encoding="utf-8"))
OTHER_ID = "b1b2b3b4b5b6b7b8"
OWN_ID = "c0ffee0000000001"
LEAKS = (SENTINEL_HOST, "203.0.113.77", "buildhost")


# --- helpers ---------------------------------------------------------------------------------


def _folder(root: Path, machines: dict[str, str | None], writer: str | None = None, families: str = "") -> Path:
    """A results folder as 0.1.0 left it: `modelroom.toml` with these rows, a comment of the user's own."""
    root.mkdir(parents=True, exist_ok=True)
    lines = ["# written by hand, kept by the sweep", "schema_version = 3", "packagers = []", "publishers = []", ""]
    for key, profile in machines.items():
        lines += [f"[machines.{key}]", "reserve_vram_gib = 1.0", f"writer = {'true' if key == writer else 'false'}"]
        lines += [f'profile = "{profile}"', ""] if profile else [""]
    lines += ["[paths]", 'state = "state"', 'markdown = "docs/models.md"', "", families]
    config = root / "modelroom.toml"
    config.write_bytes("\n".join(lines).encode("utf-8"))
    return config


def _profile(root: Path, profile_id: str, name: str, machine_class: str = "unknown") -> dict:
    changes = {"machine_class": machine_class, "machine_class_source": "unknown" if machine_class == "unknown" else "entered"}
    data = v2_profile(profile_id, display_name=name).model_copy(update=changes).model_dump(mode="json")
    _write(root / "state" / "hardware" / f"{profile_id}.json", data)
    return data


def _record(**changes) -> MeasurementRecord:
    return MeasurementRecord.model_validate({**EXPORT["measurements"][0], **changes})


def _leaky_record(profile_id: str = PROFILE_ID) -> MeasurementRecord:
    return _record(
        profile_id=profile_id,
        ollama_name=SENTINEL_MODEL,
        daemon_version=SENTINEL_VERSIONS[0],
        comparable=False,
        comparable_reason=f"/api/ps showed {SENTINEL_MODEL}, not nova:7b",
    )


def _store(root: Path, record: MeasurementRecord) -> Path:
    path = root / "state" / "measurements" / record.profile_id / f"{record.measurement_id}.json"
    return _write(path, record.model_dump(mode="json"))


def _write(path: Path, data: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((data if isinstance(data, str) else json.dumps(data, indent=2)).encode("utf-8"))
    return path


def _snapshot(root: Path, packages: list | None = None) -> None:
    unknown = {"source_revision": None, "kind": "unknown"}
    repos = {package.base_model_hf_repo for package in packages or []}
    bases = [BaseModelSpec(hf_repo=repo, publisher=repo.split("/")[0], architecture={"source_repo": repo, **unknown}) for repo in repos]
    snapshot = Snapshot(schema_version=1, run_at=NOW, areas=[], base_models=bases, packages=packages or [])
    _write(root / "state" / "modelroom.json", snapshot.model_dump(mode="json"))


def _pointer(home: Path, bindings: dict[Path, str] | None = None) -> Path:
    pointer = home / ".modelroom" / "guided.json"
    if bindings:
        _write(pointer, {"schema_version": 1, "current": None, "bindings": {str(k.resolve()): v for k, v in bindings.items()}})
    return pointer


def _machines(config: Path) -> dict:
    return tomllib.loads(config.read_text(encoding="utf-8"))["machines"]


def _display(root: Path, profile_id: str) -> str:
    return json.loads((root / "state" / "hardware" / f"{profile_id}.json").read_text(encoding="utf-8"))["display_name"]


def _clean(*folders: Path) -> None:
    for folder in folders:
        for needle in LEAKS:
            assert files_with(folder, needle) == [], (folder.name, needle)


def _old_folder(root: Path) -> Path:
    """A 0.1.0 folder: this computer's row under its host name, names and foreign text in every file."""
    config = _folder(root, {SENTINEL_HOST: PROFILE_ID, f"{SENTINEL_HOST}-2": OTHER_ID}, writer=SENTINEL_HOST)
    profile = _profile(root, PROFILE_ID, SENTINEL_HOST, "laptop")
    _profile(root, OTHER_ID, f"{SENTINEL_HOST}-2", "server")
    _store(root, _leaky_record())
    _write(root / "modelroom.toml.bak", config.read_text(encoding="utf-8"))
    _write(root / "state" / "hardware" / f"{PROFILE_ID}.json.v3.bak", profile)
    return config


# --- the rules, field by field ----------------------------------------------------------------


@pytest.mark.parametrize(
    "found, expected",
    [
        (lambda: display_name_for(PROFILE_ID, "server", "laptop-0a1b2c3d", None), "laptop-0a1b2c3d"),
        (lambda: display_name_for(PROFILE_ID, "server", SENTINEL_HOST, "gpu1"), "gpu1"),
        (lambda: display_name_for(PROFILE_ID, "server", SENTINEL_HOST, None), "server-3f9a0c21"),
        (lambda: display_name_for(PROFILE_ID, "unknown", SENTINEL_HOST, None), "computer-3f9a0c21"),
        (lambda: machine_keys({"server-0a1b2c3d": PROFILE_ID}, {}, Names())["server-0a1b2c3d"], "server-0a1b2c3d"),
        (lambda: machine_keys({"gpu1": PROFILE_ID}, {}, Names(aliases={PROFILE_ID: "gpu1"}))["gpu1"], "gpu1"),
        (lambda: machine_keys({"gpu1": OTHER_ID}, {OTHER_ID: "server"}, Names(aliases={PROFILE_ID: "gpu1"}))["gpu1"], "server-b1b2b3b4"),
        (lambda: machine_keys({SENTINEL_HOST: PROFILE_ID}, {PROFILE_ID: "laptop"}, Names())[SENTINEL_HOST], "laptop-3f9a0c21"),
        (lambda: machine_keys({SENTINEL_HOST: None}, {}, Names(this_machine=THIS_MACHINE, old=SENTINEL_HOST))[SENTINEL_HOST], THIS_MACHINE),
        (lambda: neutral_version("0.12.3"), "0.12.3"),
        (lambda: neutral_version("0.12.3-rc1"), None),
        (lambda: comparable_reason(NO_OBSERVATION), NO_OBSERVATION),
        (lambda: comparable_reason("the daemon loaded context 4096, not the ranking's 8192"), "the daemon loaded context 4096, not the ranking's 8192"),
        (lambda: comparable_reason(f"/api/ps showed {SENTINEL_MODEL}, not nova:7b"), ANOTHER_MODEL),
        (lambda: validity_reason("run 2: eval_count 12, expected 128"), "run 2: eval_count 12, expected 128"),
        (lambda: validity_reason(f"run 1: done_reason '{SENTINEL_HOST}', expected 'length'"), UNKNOWN_VALIDITY),
        (lambda: record_name("hf.co/a/b:Q4_K_M", "a/b"), "hf.co/a/b:Q4_K_M"),
        (lambda: record_name("hf.co/a/b:Q4_K_M", "a/c"), None),
        (lambda: record_name(SENTINEL_MODEL, None), None),
    ],
)
def test_each_field_keeps_only_what_the_table_allows(found, expected):
    assert found() == expected


def test_a_row_without_profile_gets_a_random_neutral_key():
    drawn = iter(["computer-00000000", "computer-11111111"])  # the first one is taken: drawn again
    keys = machine_keys({SENTINEL_HOST: None, "computer-00000000": None}, {}, Names(), new_key=lambda: next(drawn))
    assert keys == {"computer-00000000": "computer-00000000", SENTINEL_HOST: "computer-11111111"}


def test_a_version_with_any_suffix_is_null():
    for version in (*SENTINEL_VERSIONS, "0.12.3-rc1", "v0.12.3", "0.12"):
        assert neutral_record(_record(daemon_version=version)).daemon_version is None
    assert neutral_record(_record(daemon_version="0.12.3")).daemon_version == "0.12.3"


def test_a_reason_naming_another_model_becomes_the_fixed_sentence():
    record = _record(comparable=False, comparable_reason=f"/api/ps showed {SENTINEL_MODEL}, other:1b, not nova:7b")
    assert neutral_record(record).comparable_reason == ANOTHER_MODEL
    changed = _record(comparable=False, comparable_reason=f"the digest of {SENTINEL_MODEL} changed during the measurement")
    assert neutral_record(changed).comparable_reason == DIGEST_CHANGED


def test_an_unknown_reason_is_not_kept():
    assert neutral_record(_record(comparable=False, comparable_reason=f"{SENTINEL_HOST} said so")).comparable_reason == UNKNOWN_COMPARABLE
    invalid = _record(validity="invalid", validity_reason=f"run 1: done_reason '{SENTINEL_HOST}', expected 'length'")
    assert neutral_record(invalid).validity_reason == UNKNOWN_VALIDITY
    run = {**EXPORT["measurements"][0]["runs"][0], "done_reason": SENTINEL_HOST}
    foreign = _record(validity="invalid", validity_reason="run 1: x", runs=[run] * 3, tps_mean=None, tps_min=None, tps_max=None)
    assert {run.done_reason for run in neutral_record(foreign).runs} == {"unknown"}


def test_a_record_name_without_package_is_null_unless_it_is_the_hf_name_of_its_repo():
    registry = _record(ollama_name=SENTINEL_MODEL)
    assert neutral_record(registry).ollama_name is None
    assert neutral_record(registry, granite_ollama_package()).ollama_name == "granite4.2:8b"
    ref = {"content_source": "huggingface", "ollama_manifest_digest": None, "hf_repo": DEEPSEEK_REPO, "hf_revision": DEEPSEEK_SHA, "hf_file_digest": DEEPSEEK_WEIGHTS_DIGEST}
    assert neutral_record(_record(package=ref, ollama_name=DEEPSEEK_OLLAMA_NAME)).ollama_name == DEEPSEEK_OLLAMA_NAME
    assert neutral_record(_record(package=ref, ollama_name="hf.co/someone/else:Q4_K_M")).ollama_name is None


def test_this_machine_key_is_random_and_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_store, "_NEW_KEY", neutral.random_key)  # the real maker, not the suite's fixed one
    first = remote_store.this_machine(tmp_path / "a" / "remote.json")
    assert neutral.THIS_MACHINE_RE.fullmatch(first)
    assert remote_store.this_machine(tmp_path / "a" / "remote.json") == first
    assert remote_store.this_machine(tmp_path / "b" / "remote.json") != first  # 1 in 4 billion to fail


def test_the_sink_is_idempotent(tmp_path):
    record = neutral_record(_leaky_record())
    assert neutral_record(record) == record
    entries, names = {SENTINEL_HOST: PROFILE_ID, "gpu1": OTHER_ID}, Names(aliases={OTHER_ID: "gpu1"})
    keys = machine_keys(entries, {}, names)
    renamed = {keys[key]: profile for key, profile in entries.items()}
    assert machine_keys(renamed, {}, names) == {key: key for key in renamed}
    config = _old_folder(tmp_path / "old")
    sweep(config, NOW, "render", pointer_path=_pointer(tmp_path / "home"))
    before = {path: path.read_bytes() for path in config.parent.rglob("*") if path.is_file()}
    assert sweep(config, NOW, "render", pointer_path=_pointer(tmp_path / "home")).notes == []
    assert {path: path.read_bytes() for path in config.parent.rglob("*") if path.is_file()} == before


# --- the whole way, and the render of a folder of 0.1.0 --------------------------------------------


def _daemon(tags: list[dict], ps: list[dict]) -> FixtureDaemon:
    def answer(data: dict) -> Response:
        return Response(status=200, body=json.dumps(data).encode("utf-8"))

    return FixtureDaemon(
        {
            ("GET", VERSION_PATH): answer({"version": SENTINEL_VERSIONS[0]}),
            ("GET", TAGS_PATH): answer({"models": tags}),
            ("GET", PS_PATH): [answer({"models": ps})] * 4,
            ("POST", GENERATE_PATH): default_generates(),
        }
    )


def test_measure_export_share_import_render_carries_no_sentinel(tmp_path, capsys):
    """A server measured and load-tested under a host name, its export in a shared folder, imported on a
    laptop and rendered: no file of either folder, the shared folder or `remote.json` names it.

    The server runs the subcommands, so no start line names a host here; the laptop takes the file
    with `import-profile` (the shared folder's own row in the machine list is FR-S's, built after FR-N).
    """
    server, laptop, share = tmp_path / "server", tmp_path / "laptop", tmp_path / "share"
    server_config = _folder(server, {SENTINEL_HOST: None}, writer=SENTINEL_HOST)
    server_home, laptop_home = _pointer(tmp_path / "server-home"), _pointer(tmp_path / "laptop-home")
    probes = windows_probes(host=SENTINEL_HOST)
    _snapshot(server, [granite_ollama_package()])  # the server fetched the package: its identity is known there
    hardware = ["hardware", "--config", str(server_config), "--machine", SENTINEL_HOST, "--machine-class", "server"]
    assert main(hardware, probes=probes, pointer_path=server_home, now=NOW) == 0
    profile_id = read_pointer(server_home).binding_for(server.resolve())
    digest = GRANITE_MANIFEST_DIGEST.removeprefix("sha256:")
    installed = {"name": SENTINEL_MODEL, "model": SENTINEL_MODEL, "size": 5_347_929_757, "digest": digest, "details": {}}
    other = {"name": "sentinel-host:7b", "model": "sentinel-host:7b", "digest": digest, "context_length": 8192}
    daemon = _daemon([installed], [other])
    listed, _ = installed_models(daemon, NOW)
    [candidate] = installed_candidates([granite_ollama_package()], listed, lambda name: (None, "none")).candidates
    progress: list[str] = []
    record = run_load_test(daemon, candidate, default_scenario(), profile_id, probes=probes, measured_at=NOW, cpu_load=lambda: None, progress=progress.append)
    store_measurement(server / "state", record, NOW, candidate.package)
    out = share / "server.json"
    assert main(["export-profile", "--config", str(server_config), "--out", str(out)], probes=probes, pointer_path=server_home) == 0
    laptop_config = _folder(laptop, {"laptop-host": None}, writer="laptop-host")
    laptop_probes = windows_probes(host="laptop-host")
    assert main(["import-profile", str(out), "--config", str(laptop_config)], probes=laptop_probes, pointer_path=laptop_home, now=NOW) == 0
    _snapshot(laptop)
    assert main(["render", "--config", str(laptop_config)], probes=laptop_probes, pointer_path=laptop_home, now=NOW) == 0

    _clean(server, laptop, share, server_home.parent, laptop_home.parent)
    screen = capsys.readouterr()
    for needle in LEAKS:
        assert needle not in screen.out + screen.err + "\n".join(progress)
    def stored(root: Path) -> MeasurementRecord:
        return MeasurementRecord.model_validate_json(next((root / "state" / "measurements").rglob("*.json")).read_text(encoding="utf-8"))

    # The server's snapshot knows the package: its name; the laptop's does not: none, the rule without one.
    assert (stored(server).ollama_name, stored(server).daemon_version, stored(server).comparable_reason) == ("granite4.2:8b", None, ANOTHER_MODEL)
    assert (stored(laptop).ollama_name, stored(laptop).comparable_reason) == (None, ANOTHER_MODEL)
    assert "granite4.2:8b: warm-up run" in progress


def test_render_on_a_0_1_0_folder_writes_no_host_name(tmp_path):
    config = _old_folder(tmp_path / "old")
    _snapshot(config.parent)

    code = main(["render", "--config", str(config)], probes=windows_probes(host="laptop-host"), pointer_path=_pointer(tmp_path / "home"), now=NOW)

    assert code == 0
    _clean(config.parent)
    assert (config.parent / "docs" / "models.md").is_file()
    assert config.read_text(encoding="utf-8").startswith("# written by hand, kept by the sweep")


def test_render_whose_sweep_fails_writes_no_document(tmp_path, capsys):
    config = _old_folder(tmp_path / "old")
    _snapshot(config.parent)
    document = _write(config.parent / "docs" / "models.md", "the document of an earlier render\n")
    handle = acquire_lock(config.parent / "state" / "modelroom.lock", "another process", NOW)
    try:
        code = main(["render", "--config", str(config)], probes=windows_probes(host="laptop-host"), pointer_path=_pointer(tmp_path / "home"), now=NOW)
    finally:
        release_lock(handle)

    assert code == 1
    assert FAILED in capsys.readouterr().err
    assert document.read_text(encoding="utf-8") == "the document of an earlier render\n"
    assert SENTINEL_HOST in _machines(config)


# --- the folder as it is: sweep ---------------------------------------------------------------------


def test_sweep_renames_this_machines_old_key(tmp_path):
    config = _folder(tmp_path / "old", {SENTINEL_HOST: OWN_ID, "server-0a1b2c3d": OTHER_ID}, writer=SENTINEL_HOST)

    swept = sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"), hostname=SENTINEL_HOST)

    assert swept.renames == {SENTINEL_HOST: THIS_MACHINE}
    assert swept.notes == [f"machine renamed to {THIS_MACHINE} (no host names in files)"]
    assert _machines(config)[THIS_MACHINE] == {"reserve_vram_gib": 1.0, "writer": True, "profile": OWN_ID}
    assert "server-0a1b2c3d" in _machines(config)


def test_a_folder_neutralized_by_another_computer_still_fetches_here(tmp_path):
    """Codex r5-2: computer B sweeps A's folder of 0.1.0; A, bound to its profile, still fetches there."""
    config = _folder(tmp_path / "shared", {"a-host": OWN_ID}, writer="a-host")
    _profile(config.parent, OWN_ID, "a-host", "laptop")
    b_home = _pointer(tmp_path / "b")
    _write(b_home.parent / "remote.json", {"schema_version": 1, "this_machine": "computer-0b0b0b0b"})
    assert sweep(config, NOW, "render", pointer_path=b_home, hostname="b-host").renames == {"a-host": "laptop-c0ffee00"}

    a_home = _pointer(tmp_path / "a", {config.parent: OWN_ID})
    code = main(["fetch", "--config", str(config), "--machine", THIS_MACHINE], transport=build_transport({}), probes=windows_probes(host="a-host"), pointer_path=a_home, now=NOW)

    assert code == 0
    assert _machines(config) == {THIS_MACHINE: {"reserve_vram_gib": 1.0, "writer": True, "profile": OWN_ID}}


def test_this_machine_already_on_another_row_yields_to_the_binding(tmp_path):
    config = _folder(tmp_path / "old", {THIS_MACHINE: OTHER_ID, "old-host": OWN_ID})
    _profile(config.parent, OTHER_ID, "server-b1b2b3b4", "server")

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home", {config.parent: OWN_ID}))

    assert {key: entry["profile"] for key, entry in _machines(config).items()} == {THIS_MACHINE: OWN_ID, "server-b1b2b3b4": OTHER_ID}


def test_sweep_fixes_a_neutral_profile_with_an_old_key(tmp_path):
    config = _folder(tmp_path / "old", {SENTINEL_HOST: PROFILE_ID})
    profile = _profile(config.parent, PROFILE_ID, "server-3f9a0c21", "server")

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    assert list(_machines(config)) == ["server-3f9a0c21"]
    assert json.loads((config.parent / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8")) == profile


def test_sweep_fixes_an_old_profile_with_a_neutral_key(tmp_path):
    config = _folder(tmp_path / "old", {"server-3f9a0c21": PROFILE_ID})
    before = config.read_bytes()
    _profile(config.parent, PROFILE_ID, SENTINEL_HOST, "server")

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    assert config.read_bytes() == before
    assert _display(config.parent, PROFILE_ID) == "server-3f9a0c21"


def test_sweep_cleans_records_and_backups(tmp_path):
    config = _old_folder(tmp_path / "old")

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    _clean(config.parent)
    backup = tomllib.loads((config.parent / "modelroom.toml.bak").read_text(encoding="utf-8"))
    assert sorted(backup["machines"]) == sorted(_machines(config))
    assert (config.parent / "modelroom.toml.bak").read_text(encoding="utf-8").startswith("# written by hand")


def test_sweep_writes_no_new_backup(tmp_path):
    config = _old_folder(tmp_path / "old")
    names = {path.relative_to(config.parent) for path in config.parent.rglob("*")}

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    assert {path.relative_to(config.parent) for path in config.parent.rglob("*")} - names == {Path("state/modelroom.lock")}


def test_an_unreadable_backup_is_a_note_without_its_content(tmp_path):
    config = _folder(tmp_path / "old", {"server-3f9a0c21": PROFILE_ID})
    broken = config.parent / "modelroom.toml.v2.bak"
    broken.write_bytes(b"\xff\xfe [machines." + SENTINEL_HOST.encode() + b"]")

    swept = sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    assert swept.notes == ["a backup of modelroom could not be cleaned of machine names: modelroom.toml.v2.bak; delete it by hand"]
    assert broken.read_bytes().startswith(b"\xff\xfe")


@pytest.mark.parametrize("command", ["hardware", "fetch", "render", "migrate", "import-profile", "export-profile", "guided"])
def test_every_writing_command_sweeps_first(tmp_path, command, monkeypatch):
    """The sweep's rewrite of the configuration comes before the first file the command writes itself."""
    written: list[str] = []
    real = state_module.atomic_write_json, state_module.atomic_write_text
    for module in [module for name, module in sys.modules.items() if name.startswith("modelroom") and module is not None]:
        for index, name in enumerate(("atomic_write_json", "atomic_write_text")):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, lambda path, data, _real=real[index]: (written.append(Path(path).name), _real(path, data))[1])
    results = tmp_path / "results"
    config = _folder(results, {SENTINEL_HOST: PROFILE_ID}, writer=SENTINEL_HOST)
    _profile(results, PROFILE_ID, SENTINEL_HOST, "laptop")
    _snapshot(results)
    home, probes = _pointer(tmp_path / "home"), windows_probes(host=SENTINEL_HOST)
    base = ["--config", str(config)]
    argv = {
        "hardware": ["hardware", *base],
        "fetch": ["fetch", *base, "--machine", SENTINEL_HOST],
        "render": ["render", *base],
        "migrate": ["migrate", *base],
        "import-profile": ["import-profile", str(_write(tmp_path / "e.json", EXPORT)), *base],
        "export-profile": ["export-profile", *base, "--profile", PROFILE_ID, "--out", str(tmp_path / "out.json")],
    }
    if command == "guided":
        run_guided(
            FileAsker({"results": "here", "machines": [], "search": "qwen", "filter_owners": True, "select": [], "users": 1, "context": "8192"}),
            here=results, pointer_path=home, transport=build_transport(guided_transport_mapping()), probes=probes,
            daemon=offline_daemon(), servers=offline_servers(), now=NOW, out=lambda line: None,
        )
    else:
        main(argv[command], transport=build_transport({}), probes=probes, pointer_path=home, now=NOW.replace(hour=13))  # after the snapshot

    assert SENTINEL_HOST not in _machines(config)
    assert neutral.is_neutral(_display(results, PROFILE_ID))  # `import-profile` then takes the newer export's
    own = {"hardware": "guided.json", "fetch": "modelroom.json", "render": "models.json", "migrate": None,
           "import-profile": "20260923T083000Z-5c1e9a07.json", "export-profile": "out.json", "guided": "search.json"}[command]
    assert "modelroom.toml" in written and (own is None or written.index("modelroom.toml") < written.index(own)), written


def test_no_writer_bypasses_the_senke(tmp_path, monkeypatch):
    """Every writer of the package, watched: none may be handed a machine name, whatever the command."""
    results, other = tmp_path / "results", tmp_path / "other"
    config = _old_folder(results)
    _snapshot(results)
    export = _write(tmp_path / "e.json", {**EXPORT, "profile": {**EXPORT["profile"], "display_name": SENTINEL_HOST}})
    other_config = _folder(other, {"other-host": None}, writer="other-host")
    written: list[str] = []

    def watched(real):
        def write(path, content, *args):
            text = content if isinstance(content, str) else json.dumps(content)
            assert SENTINEL_HOST not in text, f"{path} would carry a machine name"
            written.append(Path(path).name)
            return real(path, content, *args)

        return write

    for module in [module for name, module in sys.modules.items() if name.startswith("modelroom") and module is not None]:
        for name in ("atomic_write_json", "atomic_write_text", "publish_new_text"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, watched(getattr(module, name)))
    home, probes = _pointer(tmp_path / "home"), windows_probes(host=SENTINEL_HOST)
    for argv in (
        ["hardware", "--config", str(config)],
        ["render", "--config", str(config)],
        ["export-profile", "--config", str(config), "--profile", OTHER_ID, "--out", str(tmp_path / "out.json")],
        ["import-profile", str(export), "--config", str(other_config)],
        ["migrate", "--config", str(other_config)],
    ):
        assert main(argv, probes=probes, pointer_path=home, now=NOW) == 0, argv
    run_guided(
        FileAsker({"results": "here", "machines": ["this-machine"], "machine_class": "laptop", "search": "qwen", "filter_owners": True, "select": [], "users": 1, "context": "8192"}),
        here=results, pointer_path=home, transport=build_transport(guided_transport_mapping()), probes=probes,
        daemon=offline_daemon(), servers=offline_servers(), now=NOW, out=lambda line: None,
    )

    assert {"modelroom.toml", "out.json", "models.md", "remote.json"} <= set(written)
    _clean(results, other)


# --- export and import ---------------------------------------------------------------------------


def test_an_export_carries_neutral_profile_and_records(tmp_path):
    config = _old_folder(tmp_path / "old")
    out = tmp_path / "out.json"

    code = main(["export-profile", "--config", str(config), "--profile", PROFILE_ID, "--out", str(out)], pointer_path=_pointer(tmp_path / "home"))

    assert code == 0
    exported = json.loads(out.read_text(encoding="utf-8"))
    assert exported["profile"]["display_name"] == "laptop-3f9a0c21"
    assert [m["comparable_reason"] for m in exported["measurements"]] == [ANOTHER_MODEL]
    _clean(tmp_path)


def test_an_import_neutralizes_before_planning(tmp_path, capsys):
    config = _folder(tmp_path / "laptop", {THIS_MACHINE: None}, writer=THIS_MACHINE)
    leaky = {**EXPORT, "profile": {**EXPORT["profile"], "display_name": SENTINEL_HOST}}
    leaky["measurements"] = [_leaky_record().model_dump(mode="json"), *EXPORT["measurements"][1:]]

    code = main(["import-profile", str(_write(tmp_path / "e.json", leaky)), "--config", str(config)], pointer_path=_pointer(tmp_path / "home"), now=NOW)

    assert code == 0
    assert _machines(config)["computer-3f9a0c21"]["profile"] == PROFILE_ID
    assert _display(config.parent, PROFILE_ID) == "computer-3f9a0c21"
    assert SENTINEL_HOST not in capsys.readouterr().out
    _clean(config.parent)


def test_the_same_record_from_a_0_1_0_export_and_a_0_2_0_export_is_unchanged_not_a_conflict(tmp_path, capsys):
    config = _folder(tmp_path / "laptop", {THIS_MACHINE: None}, writer=THIS_MACHINE)
    old = _write(tmp_path / "old.json", {**EXPORT, "measurements": [_leaky_record().model_dump(mode="json")]})
    home = _pointer(tmp_path / "home")
    assert main(["import-profile", str(old), "--config", str(config)], pointer_path=home, now=NOW) == 0
    new = _write(tmp_path / "new.json", {**EXPORT, "measurements": [neutral_record(_leaky_record()).model_dump(mode="json")]})
    capsys.readouterr()

    code = main(["import-profile", str(new), "--config", str(config)], pointer_path=home, now=NOW)

    assert code == 0
    assert f"measurement {_leaky_record().measurement_id}: unchanged (already stored)" in capsys.readouterr().out


def test_the_ssh_path_keeps_its_alias(tmp_path):
    config = _folder(tmp_path / "laptop", {THIS_MACHINE: None}, writer=THIS_MACHINE)
    home = _pointer(tmp_path / "home")
    _write(home.parent / "remote.json", {"schema_version": 1, "this_machine": THIS_MACHINE, "ssh": {PROFILE_ID: "gpu1"}})

    import_profile(config, _write(tmp_path / "e.json", EXPORT), NOW, name="gpu1", pointer_path=home)

    assert _machines(config)["gpu1"]["profile"] == PROFILE_ID
    assert _display(config.parent, PROFILE_ID) == "gpu1"
    assert sweep(config, NOW, "render", pointer_path=home).renames == {}
    assert _display(config.parent, PROFILE_ID) == "gpu1"


def test_no_export_or_render_takes_content_from_guided_json(tmp_path):
    """`guided.json` keeps the folders as the user chose them; nothing of it reaches a file of a folder."""
    config = _folder(tmp_path / "old", {THIS_MACHINE: PROFILE_ID}, writer=THIS_MACHINE)
    _profile(config.parent, PROFILE_ID, "laptop-3f9a0c21", "laptop")
    _snapshot(config.parent)
    sentinel = "\\\\sentinel-host\\results" if sys.platform == "win32" else "/sentinel-host/results"
    home = _pointer(tmp_path / "home")
    _write(home, {"schema_version": 1, "current": sentinel, "bindings": {sentinel: OTHER_ID, str(config.parent.resolve()): PROFILE_ID}})
    out = tmp_path / "share" / "out.json"

    assert main(["export-profile", "--config", str(config), "--out", str(out)], pointer_path=home) == 0
    assert main(["render", "--config", str(config)], pointer_path=home, now=NOW) == 0

    _clean(config.parent, out.parent)
